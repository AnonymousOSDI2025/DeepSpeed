# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import torch

from .builder import TorchCPUOpBuilder


class NativeZ3Builder(TorchCPUOpBuilder):
    BUILD_VAR = "DS_BUILD_NATIVE_Z3"
    NAME = "native_z3"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.compile.{self.NAME}_op'

    def sources(self):
        return ['csrc/compile/native_z3.cpp', 'csrc/compile/util.cpp']

    def libraries_args(self):
        args = super().libraries_args()
        return args

    def include_paths(self):
        return ['csrc/includes', os.path.join(torch.utils.cpp_extension.CUDA_HOME, "include")]
