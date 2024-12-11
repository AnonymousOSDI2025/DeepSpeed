// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "native_z3.h"

#define USE_C10D_NCCL

#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/cuda/nccl.h>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include <torch/csrc/distributed/c10d/SymmetricMemory.hpp>

namespace n3z {

static c10::intrusive_ptr<c10d::ProcessGroup> process_group = nullptr;

class DSParam {
public:
    DSParam(long id,
            std::vector<int64_t> ds_shape,
            at::Tensor ds_tensor,
            at::Tensor grad_buffer,
            bool persistent)
        : id_(id),
          shape_(std::move(ds_shape)),
          ds_tensor_(ds_tensor),
          grad_buffer_(grad_buffer),
          persistent_(persistent)
    {
    }

    long getId() const { return id_; }
    std::vector<int64_t> getShape() const { return shape_; }
    at::Tensor getDSTensor() const { return ds_tensor_; }
    at::Tensor getGradBuffer() const { return grad_buffer_; }
    void setPersistent(bool persistent) { persistent_ = persistent; }
    bool isPersistent() const { return persistent_; }

private:
    long id_;
    std::vector<int64_t> shape_;
    at::Tensor ds_tensor_;
    at::Tensor grad_buffer_;
    bool persistent_;
};

class GraphOpStates {
public:
    GraphOpStates() {}
    ~GraphOpStates() {}

    void registerOpNArgs(const std::string& op_name, long n_args)
    {
        op_n_args_[op_name] = n_args;
        args_counter_[op_name] = n_args;
    }

    void resetArgCounter()
    {
        // std::cout << "resetArgCounter size op_n_args_ " << op_n_args_.size() << std::endl;

        for (const auto& it : op_n_args_) {
            assert(hasKey(op_n_args_, it.first));
            args_counter_[it.first] = op_n_args_.at(it.first);
        }
    }

    void decrementArgCounter(const std::string& op_name)
    {
        // std::cout << "decrementArgCounter " << op_name << std::endl;

        assert(hasKey(args_counter_, op_name));
        if (args_counter_.at(op_name) == 0) return;
        args_counter_[op_name]--;
    }

    long getArgCounter(const std::string& op_name) const
    {
        assert(hasKey(args_counter_, op_name));
        return args_counter_.at(op_name);
    }

    bool isArgCounterZero(const std::string& op_name) const
    {
        assert(hasKey(args_counter_, op_name));
        return args_counter_.at(op_name) == 0;
    }

private:
    std::unordered_map<std::string, size_t> op_n_args_;
    std::unordered_map<std::string, size_t> args_counter_;
};

c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> getSymmMemWorkspace(int64_t size)
{
    c10::Device device = c10::Device(c10::kCUDA, c10::cuda::current_device());
    std::vector<int64_t> sizes = {size};
    std::vector<int64_t> strides = {1};
    at::Tensor sym_mem_ws = c10d::symmetric_memory::empty_strided_p2p(
        {size}, {1}, c10::ScalarType::Byte, device, process_group->getGroupName(), std::nullopt);
    return c10d::symmetric_memory::rendezvous(sym_mem_ws);
}

class DSParamRegistry {
public:
    DSParamRegistry() {}
    ~DSParamRegistry() {}

    void registerParam(long ds_id,
                       const std::vector<int64_t>& ds_shape,
                       at::Tensor ds_tensor,
                       at::Tensor grad_buffer,
                       bool persistent)
    {
        grad_buffer.zero_();
        params_.emplace(ds_id, DSParam(ds_id, ds_shape, ds_tensor, grad_buffer, persistent));
    }

    void registerGatheredParam(long ds_id, at::Tensor ds_tensor)
    {
        gathered_params_.emplace(ds_id, ds_tensor);
    }

    void unregisterGatheredParam(long ds_id)
    {
        assert(hasKey(gathered_params_, ds_id));
        gathered_params_.erase(ds_id);
    }

    const std::unordered_map<long, DSParam>& getParams() const { return params_; }

    const DSParam& getParam(long ds_id) const { return params_.at(ds_id); }
    const size_t getNumParams() const { return params_.size(); }
    const at::Tensor& getGatheredParam(long ds_id) const
    {
        assert(hasKey(gathered_params_, ds_id));
        return gathered_params_.at(ds_id);
    }
    bool hasGatheredParam(long ds_id) const { return hasKey(gathered_params_, ds_id); }
    void setPersistent(long ds_id, bool persistent) { params_.at(ds_id).setPersistent(persistent); }

private:
    std::unordered_map<long, DSParam> params_;
    std::unordered_map<long, at::Tensor> gathered_params_;
};

class ReduceTask {
public:
    ReduceTask(long ds_id, at::Tensor grad, at::Tensor send_buf)
        : ds_id_(ds_id), grad_(std::move(grad)), send_buf_(std::move(send_buf))
    {
    }

    long getDSId() const { return ds_id_; }
    at::Tensor getSendBuf() const { return send_buf_; }

private:
    long ds_id_;
    at::Tensor grad_;
    at::Tensor send_buf_;
};

class ReduceBucket {
public:
    ReduceBucket(int64_t size, at::ScalarType scalar_type) : size_(size), scalar_type_(scalar_type)
    {
        buffer_ = torch::empty({size}, at::TensorOptions().dtype(scalar_type).device(at::kCUDA));
        offset_ = 0;
    }

    int64_t getSize() const { return size_; }
    int64_t getOffset() const { return offset_; }
    at::Tensor getBuffer() const { return buffer_; }
    at::ScalarType getScalarType() const { return scalar_type_; }

    void reserve(int64_t size)
    {
        if (size > size_) {
            buffer_ =
                torch::empty({size}, at::TensorOptions().dtype(scalar_type_).device(at::kCUDA));
            size_ = size;
        }
    }

    at::Tensor allocate(int64_t numel)
    {
        if (offset_ + numel > size_) {
            throw std::runtime_error("Buffer size exceeds the reduce bucket size");
        }

        at::Tensor result = buffer_.index({torch::indexing::Slice(offset_, offset_ + numel)});
        offset_ += numel;
        return result;
    }

    bool shouldFlush(int64_t numel) { return offset_ > 0 && offset_ + numel > size_; }

    void reset() { offset_ = 0; }

private:
    int64_t size_;
    int64_t offset_;
    at::Tensor buffer_;
    at::ScalarType scalar_type_;
};

class DoubleBufferedReduceBucket {
public:
    DoubleBufferedReduceBucket(int64_t initial_bucket_size, bool enable_double_buffer)
        : initial_bucket_size_(initial_bucket_size), enable_double_buffer_(enable_double_buffer)
    {
    }

    void swap(at::ScalarType scalar_type,
              at::cuda::CUDAStream rs_stream,
              at::cuda::CUDAStream copy_stream)
    {
        assert(hasKey(current_buffer_, scalar_type));
        assert(hasKey(current_buffer_events_, scalar_type));

        current_buffer_.at(scalar_type)->reset();
        current_buffer_events_.at(scalar_type)->record(rs_stream);

        if (enable_double_buffer_) {
            assert(hasKey(shadow_buffer_, scalar_type));
            assert(hasKey(shadow_buffer_events_, scalar_type));

            auto tmp = current_buffer_.at(scalar_type);
            current_buffer_[scalar_type] = shadow_buffer_.at(scalar_type);
            shadow_buffer_[scalar_type] = tmp;

            auto tmp_event = current_buffer_events_.at(scalar_type);
            current_buffer_events_[scalar_type] = shadow_buffer_events_.at(scalar_type);
            shadow_buffer_events_[scalar_type] = tmp_event;
        }
    }

    std::shared_ptr<ReduceBucket> getBuffer(at::ScalarType scalar_type)
    {
        if (!hasKey(current_buffer_, scalar_type)) {
            current_buffer_[scalar_type] =
                std::make_shared<ReduceBucket>(initial_bucket_size_, scalar_type);
            current_buffer_events_[scalar_type] =
                std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);

            if (enable_double_buffer_) {
                shadow_buffer_[scalar_type] =
                    std::make_shared<ReduceBucket>(initial_bucket_size_, scalar_type);
                shadow_buffer_events_[scalar_type] =
                    std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);
            }
        }

        return current_buffer_.at(scalar_type);
    }

    std::shared_ptr<at::cuda::CUDAEvent> getEvent(at::ScalarType scalar_type)
    {
        assert(hasKey(current_buffer_events_, scalar_type));
        return current_buffer_events_.at(scalar_type);
    }

    void clear()
    {
        current_buffer_.clear();
        shadow_buffer_.clear();
        current_buffer_events_.clear();
        shadow_buffer_events_.clear();
    }

private:
    int64_t initial_bucket_size_;
    bool enable_double_buffer_;
    std::unordered_map<at::ScalarType, std::shared_ptr<ReduceBucket>> current_buffer_;
    std::unordered_map<at::ScalarType, std::shared_ptr<ReduceBucket>> shadow_buffer_;
    std::unordered_map<at::ScalarType, std::shared_ptr<at::cuda::CUDAEvent>> current_buffer_events_;
    std::unordered_map<at::ScalarType, std::shared_ptr<at::cuda::CUDAEvent>> shadow_buffer_events_;
};

ncclDataType_t get_nccl_data_type(at::ScalarType scalar_type)
{
    switch (scalar_type) {
        case at::kFloat: return ncclFloat;
        case at::kHalf: return ncclHalf;
        case at::kDouble: return ncclDouble;
        case at::kBFloat16: return ncclBfloat16;
        case at::kLong: return ncclInt64;
        case at::kInt: return ncclInt;
        case at::kChar: return ncclInt8;
        default: throw std::runtime_error("Unsupported scalar type");
    }
}

class CustomOpExecutor {
public:
    CustomOpExecutor(c10::intrusive_ptr<c10d::ProcessGroup> process_group,
                     std::shared_ptr<DSParamRegistry> param_registry,
                     std::shared_ptr<DoubleBufferedReduceBucket> reduce_buckets,
                     std::vector<long> ds_ids,
                     ncclComm_t nccl_comm,
                     at::cuda::CUDAStream ag_stream,
                     at::cuda::CUDAStream rs_stream,
                     at::cuda::CUDAStream copy_stream,
                     at::cuda::CUDAStream offload_stream,
                     at::cuda::CUDAStream reload_stream,
                     bool pre_div_reduce)
        : process_group_(process_group),
          param_registry_(std::move(param_registry)),
          reduce_buckets_(std::move(reduce_buckets)),
          ds_ids_(std::move(ds_ids)),
          nccl_comm_(nccl_comm),
          ag_stream_(ag_stream),
          rs_stream_(rs_stream),
          copy_stream_(copy_stream),
          offload_stream_(offload_stream),
          reload_stream_(reload_stream),
          pre_div_reduce_(pre_div_reduce)
    {
        for (long ds_id : ds_ids_) {
            has_acc_grad_[ds_id] = false;
            valid_[ds_id] = false;

            ag_comm_done_events_[ds_id] =
                std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);
            ag_comp_done_events_[ds_id] =
                std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);
            rs_comp_done_events_[ds_id] =
                std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);
            rs_copy_done_events_[ds_id] =
                std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);
        }
        reduce_counter_ = ds_ids_.size();
    }
    ~CustomOpExecutor() {}

    void registerOpNArgs(const std::string& op_name, long n_args, bool is_backward)
    {
        GraphOpStates& op_states = is_backward ? op_states_bwd_ : op_states_fwd_;
        op_states.registerOpNArgs(op_name, n_args);
    }

    void startForward() {}

    void endForward() {}

    void startBackward(bool update) { param_updated_ = update; }

    void endBackward()
    {
        if (param_updated_) {
            for (auto& it : has_acc_grad_) { it.second = false; }
            for (auto& it : valid_) { it.second = false; }
        }

        for (auto& it : reload_buffers_) {
            it.second.record_stream(at::cuda::getCurrentCUDAStream());
        }
        reload_buffers_.clear();
    }

    void launchAllGather(at::Tensor output_buf,
                         long ds_id,
                         c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> symm_mem)
    {
        const DSParam& param = param_registry_->getParam(ds_id);
        const at::Tensor& ds_tensor = param.getDSTensor();

        if (symm_mem == nullptr) {
            ncclResult_t result = ncclAllGather(ds_tensor.contiguous().data_ptr(),
                                                output_buf.data_ptr(),
                                                ds_tensor.numel(),
                                                get_nccl_data_type(ds_tensor.scalar_type()),
                                                nccl_comm_,
                                                ag_stream_);

            if (result != ncclSuccess) { throw std::runtime_error("NCCL AllGather failed"); }
        } else {
            at::cuda::CUDAStreamGuard guard(ag_stream_);
            int world_size = process_group_->getSize();
            int rank = process_group_->getRank();

            at::Tensor local_buf =
                symm_mem->get_buffer(rank, ds_tensor.sizes(), ds_tensor.scalar_type(), 0);
            local_buf.copy_(ds_tensor, true);

            symm_mem->barrier(0);
            auto chunks = output_buf.flatten().chunk(world_size);
            for (int step = 0; step < world_size; step++) {
                int remote_rank = (rank - step + world_size) % world_size;
                auto src_buf = symm_mem->get_buffer(
                    remote_rank, ds_tensor.sizes(), ds_tensor.scalar_type(), 0);
                chunks[remote_rank].copy_(src_buf.flatten(), true);
            }
            symm_mem->barrier(0);
        }

        param_registry_->registerGatheredParam(ds_id, output_buf);
        valid_[ds_id] = true;
    }

    at::Tensor allgatherParam(long ds_id,
                              c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> symm_mem)
    {
        assert(hasKey(valid_, ds_id));
        if (valid_.at(ds_id)) { return param_registry_->getGatheredParam(ds_id); }

        const DSParam& param = param_registry_->getParam(ds_id);
        const at::Tensor& ds_tensor = param.getDSTensor();
        at::Tensor output_buf = param_registry_->hasGatheredParam(ds_id)
                                    ? param_registry_->getGatheredParam(ds_id)
                                    : torch::empty(param.getShape(), ds_tensor.options());

        ag_comp_done_events_[ds_id]->record();
        ag_comp_done_events_[ds_id]->block(ag_stream_);

        launchAllGather(output_buf, ds_id, symm_mem);

        ag_comm_done_events_[ds_id]->record(ag_stream_);
        return output_buf;
    }

    void prefetchParamsFused(std::vector<int64_t> ds_ids,
                             c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> symm_mem)
    {
        std::vector<int64_t> invalid_ds_ids;
        for (const auto& ds_id : ds_ids) {
            assert(hasKey(valid_, ds_id));
            if (!valid_.at(ds_id)) { invalid_ds_ids.push_back(ds_id); }
        }

        std::unordered_map<long, at::Tensor> output_bufs;
        for (long ds_id : invalid_ds_ids) {
            const DSParam& param = param_registry_->getParam(ds_id);
            if (param_registry_->hasGatheredParam(ds_id)) {
                output_bufs[ds_id] = param_registry_->getGatheredParam(ds_id);
            } else {
                output_bufs[ds_id] = torch::empty(param.getShape(), param.getDSTensor().options());
            }
        }

        for (long ds_id : invalid_ds_ids) {
            ag_comp_done_events_[ds_id]->record();
            ag_comp_done_events_[ds_id]->block(ag_stream_);
        }

        ncclGroupStart();
        for (long ds_id : invalid_ds_ids) {
            assert(hasKey(output_bufs, ds_id));
            launchAllGather(output_bufs.at(ds_id), ds_id, symm_mem);
        }
        ncclGroupEnd();

        for (long ds_id : invalid_ds_ids) { ag_comm_done_events_[ds_id]->record(ag_stream_); }
    }

    at::Tensor releaseParam(at::Tensor v, long ds_id)
    {
        const DSParam& param = param_registry_->getParam(ds_id);

        if (!param.isPersistent()) {
            at::Tensor gathered_param = param_registry_->getGatheredParam(ds_id);

            if (gathered_param.defined()) {  // gathered param is undefined while profiling
                const auto options = gathered_param.options();
                at::Tensor empty_buffer = torch::empty({0}, options);
                gathered_param.set_data(empty_buffer);
            }

            param_registry_->unregisterGatheredParam(ds_id);
            valid_[ds_id] = false;
        }

        return v;
    }

    at::Tensor waitAllgather(at::Tensor v,
                             long ds_id,
                             const std::string& user,
                             long n_args,
                             bool is_backward)
    {
        GraphOpStates& op_states = is_backward ? op_states_bwd_ : op_states_fwd_;

        op_states.decrementArgCounter(user);

        if (op_states.isArgCounterZero(user)) {
            assert(hasKey(ag_comm_done_events_, ds_id));
            ag_comm_done_events_[ds_id]->block(at::cuda::getCurrentCUDAStream());
            op_states_fwd_.resetArgCounter();
        }

        return v;
    }

    at::Tensor reduceGrad(at::Tensor grad_tensor, long ds_id)
    {
        int world_size = process_group_->getSize();
        const DSParam& param = param_registry_->getParam(ds_id);
        const auto scalar_type = grad_tensor.scalar_type();
        std::shared_ptr<ReduceBucket> reduce_bucket = reduce_buckets_->getBuffer(scalar_type);

        auto comp_stream = at::cuda::getCurrentCUDAStream();

        if (reduce_bucket->shouldFlush(grad_tensor.numel())) {
            flushReduceBucket(scalar_type);

            // reduce_bucket is swapped in flushReduceBucket if double buffering is enabled
            reduce_bucket = reduce_buckets_->getBuffer(scalar_type);
        }

        if (grad_tensor.numel() > reduce_bucket->getSize()) {
            // extend buckets
            at::cuda::stream_synchronize(rs_stream_);
            reduce_bucket->reserve(grad_tensor.numel());
        }

        at::Tensor reduce_in_buffer = reduce_bucket->allocate(grad_tensor.numel());

        // This ensures the order of reduce_scatter -> copy
        // Without this block, copy may start while reduce_scatter is still running
        reduce_buckets_->getEvent(scalar_type)->block(comp_stream);
        auto copy_src = grad_tensor.contiguous().view({-1});
        // keep references to copy src
        reduce_tasks_[scalar_type].emplace_back(ds_id, copy_src, reduce_in_buffer);

        // computation must be done before copy
        rs_comp_done_events_[ds_id]->record(comp_stream);
        rs_comp_done_events_[ds_id]->block(copy_stream_);
        {
            at::cuda::CUDAStreamGuard guard(copy_stream_);
            reduce_in_buffer.copy_(copy_src, true);
            rs_copy_done_events_[ds_id]->record(copy_stream_);
        }

        reduce_counter_--;

        if (reduce_counter_ == 0) {
            flushAllReduceBuckets();

            reduce_counter_ = ds_ids_.size();

            // This synchronization ensures all of reduce calls are done before optimizer's step.
            at::cuda::stream_synchronize(rs_stream_);

            endBackward();
        }

        return at::Tensor();
    }

    at::Tensor offloadTensor(at::Tensor tensor, long id)
    {
        if (!hasKey(offload_events_, id)) {
            offload_events_[id] = std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);
            offload_comp_done_events_[id] =
                std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);

            const auto options = at::TensorOptions().pinned_memory(true).device(torch::kCPU);
            offload_buffers_[id] = at::empty_like(tensor, options);
        }

        offload_comp_done_events_[id]->record();
        offload_comp_done_events_[id]->block(offload_stream_);
        {
            at::cuda::CUDAStreamGuard guard(offload_stream_);
            offload_buffers_.at(id).copy_(tensor, true);
        }

        tensor.record_stream(offload_stream_);

        offload_events_[id]->record(offload_stream_);
        assert(hasKey(offload_buffers_, id));
        return offload_buffers_.at(id);
    }

    at::Tensor reloadTensor(at::Tensor tensor, long id)
    {
        if (!hasKey(reload_events_, id)) {
            reload_events_[id] = std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);
        }

        assert(hasKey(offload_buffers_, id));
        offload_events_[id]->block(reload_stream_);

        at::Tensor ten;
        {
            at::cuda::CUDAStreamGuard guard(reload_stream_);

            assert(hasKey(offload_buffers_, id));
            at::Tensor buf = offload_buffers_.at(id);
            const auto options = at::TensorOptions().device(torch::kCUDA);
            ten = at::empty_like(buf, options);
            ten.copy_(buf, true);

            reload_buffers_[id] = ten;
        }

        reload_events_[id]->record(reload_stream_);
        return ten;
    }

    at::Tensor waitOffload(at::Tensor tensor, long id)
    {
        assert(hasKey(offload_events_, id));
        offload_events_[id]->block(at::cuda::getCurrentCUDAStream());

        assert(hasKey(offload_buffers_, id));
        return offload_buffers_.at(id);
    }

    at::Tensor waitReload(at::Tensor tensor, long id)
    {
        assert(hasKey(reload_events_, id));
        reload_events_[id]->block(at::cuda::getCurrentCUDAStream());

        assert(hasKey(reload_buffers_, id));
        auto ten = reload_buffers_.at(id);

        // We can't release here because the tensor is still being used
        // We will need "freeReloadedTensor" after the last user of the tensor to call
        // ".record_stream". As it is a bit complicated, we clear the buffer and do at the end of
        // the backward pass for now. reload_buffers_.erase(id);
        return ten;
    }

    bool hasReloadBuffer(long id) { return hasKey(reload_buffers_, id); }

    void invalidateGatheredParam(long ds_id)
    {
        if (hasKey(valid_, ds_id)) { valid_[ds_id] = false; }
    }

private:
    c10::intrusive_ptr<c10d::ProcessGroup> process_group_;
    std::shared_ptr<DSParamRegistry> param_registry_;
    std::shared_ptr<DoubleBufferedReduceBucket> reduce_buckets_;
    std::vector<long> ds_ids_;
    ncclComm_t nccl_comm_;
    at::cuda::CUDAStream ag_stream_;
    at::cuda::CUDAStream rs_stream_;
    at::cuda::CUDAStream copy_stream_;
    at::cuda::CUDAStream offload_stream_;
    at::cuda::CUDAStream reload_stream_;
    GraphOpStates op_states_fwd_ = GraphOpStates();
    GraphOpStates op_states_bwd_ = GraphOpStates();

    std::unordered_map<long, std::shared_ptr<at::cuda::CUDAEvent>> ag_comp_done_events_;
    std::unordered_map<long, std::shared_ptr<at::cuda::CUDAEvent>> ag_comm_done_events_;
    std::unordered_map<long, std::shared_ptr<at::cuda::CUDAEvent>> rs_comp_done_events_;
    std::unordered_map<long, std::shared_ptr<at::cuda::CUDAEvent>> rs_copy_done_events_;

    std::unordered_map<long, std::shared_ptr<at::cuda::CUDAEvent>> offload_events_;
    std::unordered_map<long, std::shared_ptr<at::cuda::CUDAEvent>> offload_comp_done_events_;
    std::unordered_map<long, std::shared_ptr<at::cuda::CUDAEvent>> reload_events_;
    std::unordered_map<long, at::Tensor> offload_buffers_;

    std::unordered_map<long, at::Tensor> reload_buffers_;

    size_t reduce_counter_ = 0;
    bool param_updated_ = false;
    std::unordered_map<at::ScalarType, std::vector<ReduceTask>> reduce_tasks_;
    std::unordered_map<long, bool> has_acc_grad_;
    std::unordered_map<long, bool> valid_;
    bool pre_div_reduce_;

    void flushReduceBucket(at::ScalarType scalar_type)
    {
        if (!hasKey(reduce_tasks_, scalar_type)) { return; }

        int64_t tmp_recv_numel = 0;
        for (const ReduceTask& t : reduce_tasks_.at(scalar_type)) {
            auto copy_done_event = rs_copy_done_events_.at(t.getDSId());
            copy_done_event->block(rs_stream_);

            if (has_acc_grad_.at(t.getDSId())) {
                tmp_recv_numel += param_registry_->getParam(t.getDSId()).getGradBuffer().numel();
            }
        }

        at::Tensor tmp_recv_buf = at::Tensor();
        if (tmp_recv_numel > 0) {
            at::cuda::CUDAStreamGuard guard(rs_stream_);
            tmp_recv_buf = torch::empty({tmp_recv_numel},
                                        at::TensorOptions().dtype(scalar_type).device(at::kCUDA));
        }

        ncclGroupStart();
        int64_t offset = 0;
        for (const ReduceTask& t : reduce_tasks_.at(scalar_type)) {
            auto recv_buf = param_registry_->getParam(t.getDSId()).getGradBuffer();

            bool acc_grad = has_acc_grad_.at(t.getDSId());

            if (acc_grad) {
                recv_buf =
                    tmp_recv_buf.index({torch::indexing::Slice(offset, offset + recv_buf.numel())});
            }

            ncclRedOp_t op = pre_div_reduce_ ? ncclSum : ncclAvg;
            if (pre_div_reduce_) {
                at::cuda::CUDAStreamGuard guard(rs_stream_);
                t.getSendBuf().div_(process_group_->getSize());
            }
            ncclResult_t result = ncclReduceScatter(t.getSendBuf().data_ptr(),
                                                    recv_buf.data_ptr(),
                                                    recv_buf.numel(),
                                                    get_nccl_data_type(scalar_type),
                                                    op,
                                                    nccl_comm_,
                                                    rs_stream_);
            if (result != ncclSuccess) { throw std::runtime_error("NCCL ReduceScatter failed"); }

            if (acc_grad) { offset += recv_buf.numel(); }
        }
        ncclGroupEnd();

        {
            at::cuda::CUDAStreamGuard guard(rs_stream_);
            int64_t offset = 0;
            for (const ReduceTask& t : reduce_tasks_.at(scalar_type)) {
                bool acc_grad = has_acc_grad_.at(t.getDSId());

                if (acc_grad) {
                    auto recv_buf = param_registry_->getParam(t.getDSId()).getGradBuffer();
                    recv_buf.add_(tmp_recv_buf.index(
                        {torch::indexing::Slice(offset, offset + recv_buf.numel())}));
                    offset += recv_buf.numel();
                }
                has_acc_grad_[t.getDSId()] = true;
            }
        }

        reduce_buckets_->swap(scalar_type, rs_stream_, copy_stream_);

        // Not very sure if this is necessary
        // Want to prevent grad tensor from being released before the copy is done
        auto comp_stream = at::cuda::getCurrentCUDAStream();
        for (const ReduceTask& t : reduce_tasks_.at(scalar_type)) {
            auto copy_done_event = rs_copy_done_events_.at(t.getDSId());
            copy_done_event->block(comp_stream);
        }
        reduce_tasks_[scalar_type].clear();

        if (tmp_recv_numel > 0) { tmp_recv_buf.record_stream(rs_stream_); }
    }

    void flushAllReduceBuckets()
    {
        for (const auto& it : reduce_tasks_) { flushReduceBucket(it.first); }
    }
};

static std::shared_ptr<DSParamRegistry> param_registry;
static std::unordered_map<long, std::shared_ptr<CustomOpExecutor>> executors;
std::shared_ptr<DoubleBufferedReduceBucket> reduce_buckets = nullptr;
c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> symm_mem = nullptr;

static at::cuda::CUDAStream ag_stream = at::cuda::getStreamFromPool(true);
static at::cuda::CUDAStream rs_stream = at::cuda::getStreamFromPool(true);
static at::cuda::CUDAStream copy_stream = at::cuda::getStreamFromPool(true);
static at::cuda::CUDAStream offload_stream = at::cuda::getStreamFromPool(true);
static at::cuda::CUDAStream reload_stream = at::cuda::getStreamFromPool(true);
static ncclComm_t nccl_comm;
static bool use_symm_mem;
static bool profile = false;
static bool pre_div_reduce = true;

std::vector<int64_t> sizes_to_int_vector(at::IntArrayRef sizes)
{
    std::vector<int64_t> result;
    for (int i = 0; i < sizes.size(); i++) { result.push_back(sizes[i]); }
    return result;
}

void lazy_init_symm_memory()
{
    if (use_symm_mem && !symm_mem) {
        int64_t max_param_size = 0;
        for (const auto& it : param_registry->getParams()) {
            int64_t size = it.second.getDSTensor().numel() * it.second.getDSTensor().element_size();
            if (size > max_param_size) { max_param_size = size; }
        }
        symm_mem = getSymmMemWorkspace(max_param_size);
    }
}

void enable_profiling(bool enable) { profile = enable; }

bool is_profiling() { return profile; }

void register_graph(long graph_id, const std::vector<long>& ds_ids)
{
    executors[graph_id] = std::make_shared<CustomOpExecutor>(process_group,
                                                             param_registry,
                                                             reduce_buckets,
                                                             ds_ids,
                                                             nccl_comm,
                                                             ag_stream,
                                                             rs_stream,
                                                             copy_stream,
                                                             offload_stream,
                                                             reload_stream,
                                                             pre_div_reduce);
}

void register_graph_ops(long graph_id,
                        const std::vector<std::string>& op_names,
                        const std::vector<long>& n_args)
{
    assert(op_names.size() == n_args.size());
    for (int i = 0; i < op_names.size(); i++) {
        executors[graph_id]->registerOpNArgs(op_names[i], n_args[i], false);
    }
}

void register_bwd_graph_ops(long graph_id,
                            const std::vector<std::string>& op_names,
                            const std::vector<long>& n_args)
{
    assert(hasKey(executors, graph_id));
    for (int i = 0; i < op_names.size(); i++) {
        executors[graph_id]->registerOpNArgs(op_names[i], n_args[i], true);
    }
}

void init(c10::intrusive_ptr<c10d::ProcessGroup> pg,
          int64_t initial_reduce_bucket_size,
          bool enable_double_buffer,
          bool _use_symm_mem)
{
    process_group = pg;

    ncclUniqueId ncclID;
    ncclGetUniqueId(&ncclID);

    // ProcessGroup doesn't have an API to get the CUDA stream for comm calls.
    // So we create a NCCL communicator and call NCCL APIs directly.
    auto vec = std::vector<uint8_t>(reinterpret_cast<uint8_t*>(&ncclID),
                                    reinterpret_cast<uint8_t*>(&ncclID) + NCCL_UNIQUE_ID_BYTES);
    auto device = torch::Device(torch::kCUDA);
    at::Tensor tensor = torch::from_blob(vec.data(), {static_cast<long>(vec.size())}, torch::kUInt8)
                            .to(torch::Device(torch::kCUDA));
    std::vector<at::Tensor> bcast_input = {tensor};

    process_group->broadcast(bcast_input, c10d::BroadcastOptions())->wait();

    // create a new nccl communicator
    std::memcpy(&ncclID, tensor.to(torch::Device(torch::kCPU)).data_ptr(), NCCL_UNIQUE_ID_BYTES);
    ncclCommInitRank(&nccl_comm, process_group->getSize(), ncclID, process_group->getRank());

    param_registry = std::make_shared<DSParamRegistry>();
    reduce_buckets = std::make_shared<DoubleBufferedReduceBucket>(initial_reduce_bucket_size,
                                                                  enable_double_buffer);
    use_symm_mem = _use_symm_mem;
}

void reset()
{
    executors.clear();
    // We keep the buckets for memory estimation
    // reduce_buckets->clear();
}

void cleanup()
{
    reset();

    ncclCommDestroy(nccl_comm);
    process_group = nullptr;
    symm_mem = nullptr;
}

void register_param(long ds_id,
                    const std::vector<int64_t>& ds_shape,
                    at::Tensor ds_tensor,
                    at::Tensor grad_buffer,
                    bool persistent)
{
    param_registry->registerParam(ds_id, ds_shape, ds_tensor, grad_buffer, persistent);
    if (persistent) { param_registry->registerGatheredParam(ds_id, ds_tensor); }
}

void set_persistent(long ds_id) { param_registry->setPersistent(ds_id, true); }

at::Tensor allgather_param(at::Tensor param_tensor, long graph_id, long ds_id)
{
    return executors[graph_id]->allgatherParam(ds_id, symm_mem);
}

void prefetch_params_fused(long graph_id,
                           const std::vector<at::Tensor> params,
                           const std::vector<long>& ds_ids)
{
    executors[graph_id]->prefetchParamsFused(ds_ids, symm_mem);
}

// for profiling
void invalidate_gathered_param(long ds_id)
{
    const DSParam& param = param_registry->getParam(ds_id);
    if (param.isPersistent()) { return; }

    param_registry->unregisterGatheredParam(ds_id);
    param_registry->registerGatheredParam(ds_id, at::Tensor());

    for (auto& it : executors) { it.second->invalidateGatheredParam(ds_id); }
}

void clear_all_gathered_params()
{
    for (const auto& it : param_registry->getParams()) {
        long ds_id = it.first;
        const DSParam& param = param_registry->getParam(ds_id);
        if (param.isPersistent()) { continue; }
        if (param_registry->hasGatheredParam(ds_id)) {
            param_registry->unregisterGatheredParam(ds_id);
            for (auto& it : executors) { it.second->invalidateGatheredParam(ds_id); }
        }
    }
}

at::Tensor allgather_param_meta(at::Tensor param_tensor, long graph_id, long ds_id)
{
    const DSParam& param = param_registry->getParam(ds_id);
    auto options = param.getDSTensor().options().device(c10::kMeta);
    at::Tensor output_buf = torch::empty(param.getShape(), options);
    return output_buf;
}

at::Tensor release_param(at::Tensor v, long graph_id, long ds_id)
{
    return executors[graph_id]->releaseParam(v, ds_id);
}

at::Tensor release_param_meta(at::Tensor v, long graph_id, long ds_id) { return v; }

at::Tensor wait_allgather(at::Tensor v,
                          long graph_id,
                          long ds_id,
                          const std::string& user,
                          long n_args,
                          bool is_backward)
{
    executors[graph_id]->waitAllgather(v, ds_id, user, n_args, is_backward);
    return v;
}

at::Tensor wait_allgather_meta(at::Tensor v,
                               long graph_id,
                               long ds_id,
                               const std::string& user,
                               long n_args,
                               bool is_backward)
{
    return v;
}

at::Tensor reduce_grad(at::Tensor grad_tensor, long graph_id, long ds_id)
{
    if (!profile) { executors[graph_id]->reduceGrad(grad_tensor, ds_id); }
    return at::Tensor();
}

void free_tensors(std::vector<at::Tensor> tensors)
{
    if (!profile) {
        for (auto& tensor : tensors) {
            if (tensor.is_cuda()) {
                tensor.record_stream(at::cuda::getCurrentCUDAStream());
                tensor.set_data(torch::empty({0}, tensor.options()));
            }
        }
    }
}

at::Tensor reduce_grad_meta(at::Tensor grad_tensor, long graph_id, long ds_id)
{
    return at::Tensor();
}

at::Tensor offload_tensor(at::Tensor tensor, long graph_id, long id)
{
    // auto dims = tensor.sizes();
    // std::cout << "offload_tensor graph_id=" << graph_id << " id=" << id
    //     << " dim=" << join_as_str(dims, ",") << std::endl;
    return executors[graph_id]->offloadTensor(tensor, id);
}

at::Tensor reload_tensor(at::Tensor tensor, long graph_id, long id)
{
    // auto dims = tensor.sizes();
    // std::cout << "reload_tensor graph_id=" << graph_id << " id=" << id
    //     << " dim=" << join_as_str(dims, ",") << std::endl;
    return executors[graph_id]->reloadTensor(tensor, id);
}

at::Tensor wait_offload(at::Tensor tensor, long graph_id, long id)
{
    return executors[graph_id]->waitOffload(tensor, id);
}

at::Tensor wait_reload(at::Tensor tensor, long graph_id, long id)
{
    if (profile && !executors[graph_id]->hasReloadBuffer(id)) { return tensor; }

    return executors[graph_id]->waitReload(tensor, id);
}

void start_forward()
{
    lazy_init_symm_memory();
    for (auto& it : executors) { it.second->startForward(); }
}

void end_forward()
{
    for (auto& it : executors) { it.second->endForward(); }
}

void start_backward(bool update)
{
    for (auto& it : executors) { it.second->startBackward(update); }
}

void end_backward()
{
    for (auto& it : executors) { it.second->endBackward(); }
}

at::Tensor test_call(at::Tensor a)
{
    std::cout << "test_call" << std::endl;
    return a;
}

}  // namespace n3z

TORCH_LIBRARY(native_z3, m)
{
    m.def("allgather_param(Tensor a, int graph_id, int id) -> Tensor");
    m.def("prefetch_params_fused(int graph_id, Tensor[] params, int[] ids) -> ()");
    m.def("release_param(Tensor a, int graph_id, int id) -> Tensor");
    m.def(
        "wait_allgather(Tensor a, int graph_id, int id, str user, int n_args, bool bwd) -> Tensor");
    m.def("reduce_grad(Tensor a, int graph_id, int id) -> Tensor");
    m.def("free_tensors(Tensor[] a) -> ()");
    m.def("offload_tensor(Tensor a, int id, int id) -> Tensor");
    m.def("reload_tensor(Tensor a, int id, int id) -> Tensor");
    m.def("wait_offload(Tensor a, int id, int id) -> Tensor");
    m.def("wait_reload(Tensor a, int id, int id) -> Tensor");

    m.def("test_call(Tensor a) -> Tensor");
}

TORCH_LIBRARY_IMPL(native_z3, CPU, m)
{
    m.impl("allgather_param", &n3z::allgather_param);
    m.impl("prefetch_params_fused", &n3z::prefetch_params_fused);
    m.impl("release_param", &n3z::release_param);
    m.impl("wait_allgather", &n3z::wait_allgather);
    m.impl("reduce_grad", &n3z::reduce_grad);
    m.impl("free_tensors", &n3z::free_tensors);
    m.impl("offload_tensor", &n3z::offload_tensor);
    m.impl("reload_tensor", &n3z::reload_tensor);
    m.impl("wait_offload", &n3z::wait_offload);
    m.impl("wait_reload", &n3z::wait_reload);

    m.impl("test_call", &n3z::test_call);
}

TORCH_LIBRARY_IMPL(native_z3, CUDA, m)
{
    m.impl("allgather_param", &n3z::allgather_param);
    m.impl("prefetch_params_fused", &n3z::prefetch_params_fused);
    m.impl("release_param", &n3z::release_param);
    m.impl("wait_allgather", &n3z::wait_allgather);
    m.impl("reduce_grad", &n3z::reduce_grad);
    m.impl("free_tensors", &n3z::free_tensors);
    m.impl("offload_tensor", &n3z::offload_tensor);
    m.impl("reload_tensor", &n3z::reload_tensor);
    m.impl("wait_offload", &n3z::wait_offload);
    m.impl("wait_reload", &n3z::wait_reload);

    m.impl("test_call", &n3z::test_call);
}

TORCH_LIBRARY_IMPL(native_z3, Meta, m)
{
    m.impl("allgather_param", &n3z::allgather_param_meta);
    m.impl("release_param", &n3z::release_param_meta);
    m.impl("wait_allgather", &n3z::wait_allgather_meta);
    m.impl("reduce_grad", &n3z::reduce_grad_meta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("register_param", &n3z::register_param, "Register a parameter");
    m.def("set_persistent", &n3z::set_persistent, "Set persistent flag for a parameter");
    m.def("enable_profiling", &n3z::enable_profiling, "Enable profiling");
    m.def("is_profiling", &n3z::is_profiling, "Check if profiling is enabled");
    m.def("init", &n3z::init, "Set the process group");
    m.def("cleanup", &n3z::cleanup, "Cleanup the process group");
    m.def("register_graph", &n3z::register_graph, "Register graph with a list of ds parameter ids");
    m.def("register_graph_ops",
          &n3z::register_graph_ops,
          "Register the number of arguments for an op");
    m.def("register_bwd_graph_ops",
          &n3z::register_bwd_graph_ops,
          "Register the number of arguments for a backward op");
    m.def("start_forward", &n3z::start_forward, "Start forward pass");
    m.def("end_forward", &n3z::end_forward, "End forward pass");
    m.def("start_backward", &n3z::start_backward, "Start backward pass");
    m.def("end_backward", &n3z::end_backward, "End backward pass");
    m.def("reset", &n3z::reset, "Reset the state");
    m.def(
        "invalidate_gathered_param", &n3z::invalidate_gathered_param, "Invalidate gathered param");
    m.def(
        "clear_all_gathered_params", &n3z::clear_all_gathered_params, "Clear all gathered params");
}
