// From
// https://github.com/NVIDIA/TransformerEngine/blob/e7261e116d3a27c333b8d8e3972dbea20288b101/transformer_engine/pytorch/csrc/extensions.h
#pragma once
#include "common.h"

namespace my_sputnik {

#define CONCAT_ID_(prefix, id) prefix##id
#define CONCAT_ID(prefix, id) CONCAT_ID_(prefix, id)
#define DECLARE_SPMM_KERNEL_LAUNCHER(name, id)                                \
  cudaError_t CONCAT_ID(name, id)(                                            \
      int m, int k, int n, int nonzeros, const int* __restrict__ row_indices, \
      const float* __restrict__ values, const int* __restrict__ row_offsets,  \
      const int* __restrict__ column_indices,                                 \
      const float* __restrict__ dense_matrix,                                 \
      float* __restrict__ output_matrix, cudaStream_t stream, int batch_size);

#define DECLARE_SDDMM_KERNEL_LAUNCHER(name, id)                                \
  cudaError_t CONCAT_ID(name, id)(                                             \
      int m, int k, int n, int nonzeros, const int* __restrict__ row_indices,  \
      const int* __restrict__ row_offsets,                                     \
      const int* __restrict__ column_indices,                                  \
      const float* __restrict__ lhs_matrix,                                    \
      const float* __restrict__ rhs_matrix, float* __restrict__ output_values, \
      cudaStream_t stream, int batch_size);

DECLARE_SPMM_KERNEL_LAUNCHER(CudaSpmm, 2)
DECLARE_SPMM_KERNEL_LAUNCHER(CudaSpmm, 3)
DECLARE_SDDMM_KERNEL_LAUNCHER(CudaSddmm, 2)
}  // namespace my_sputnik