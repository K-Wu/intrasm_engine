// From
// https://github.com/NVIDIA/TransformerEngine/blob/e7261e116d3a27c333b8d8e3972dbea20288b101/transformer_engine/pytorch/csrc/common.h
#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/native/DispatchStub.h>
#include <c10/macros/Macros.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <ATen/cuda/CUDAGraphsUtils.cuh>

// Define CUDAGraphConstructor
#include <helper_CUDAGraphConstructor.cu.h>