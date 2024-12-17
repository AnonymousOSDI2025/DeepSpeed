// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#define NOMINMAX  // Windows idiosyncrasy
                  // https://stackoverflow.com/questions/4913922/possible-problems-with-nominmax-on-visual-c

#include <stdio.h>
#include <torch/extension.h>

namespace n3z {

template <typename K, typename V>
static bool hasKey(const std::unordered_map<K, V>& map, const K& key)
{
    return map.find(key) != map.end();
}

template <typename T>
inline std::string to_string(const T& v)
{
    std::stringstream ss;
    ss << v;
    return ss.str();
}

template <typename L>
size_t productDim(const L& dim)
{
    size_t prod = 1;
    for (auto d : dim) { prod *= d; }
    return prod;
}

template <typename T>
std::string join_as_str(const T& v, const char* delim = ",", const size_t maxlen = 0)
{
    std::stringstream ss;

    if (!v.empty()) {
        auto it = v.begin();
        ss << to_string(*it);
        it++;
        for (; it != v.end(); ++it) {
            if (delim) ss << delim;
            ss << to_string(*it);
        }
    }

    std::string s = ss.str();
    if (maxlen > 0 && s.length() > maxlen) { s = s.substr(0, maxlen) + " ..."; }

    return "[" + s + "]";
}

template <typename T>
std::string tensorPtrToString(T* ptr, size_t size, size_t str_len = 100)
{
    std::vector<T> vals;
    for (size_t i = 0; i < size; i++) {
        vals.push_back(*ptr);
        ptr++;
    }
    return join_as_str(vals, ",", str_len);
}

std::string tensorPtrToString(void* ptr,
                              size_t size,
                              c10::ScalarType datatype,
                              size_t max_elem = 20,
                              size_t max_str_len = 100);

std::string tensorToString(const at::Tensor& t, size_t max_elem = 20, size_t max_str_len = 100);

std::string tensorDimToString(const at::Tensor& t);

at::Tensor test_call(at::Tensor param);
void register_param(long ds_id,
                    const std::vector<int64_t>& ds_shape,
                    at::Tensor ds_tensor,
                    at::Tensor grad_buffer,
                    bool persistent);

}  // namespace n3z
