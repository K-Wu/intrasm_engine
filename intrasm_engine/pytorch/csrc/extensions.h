// From
// https://github.com/NVIDIA/TransformerEngine/blob/e7261e116d3a27c333b8d8e3972dbea20288b101/transformer_engine/pytorch/csrc/extensions.h
#pragma once
#include "common.h"

namespace my_sputnik {

cudaError_t CudaSpmm2(int m, int k, int n, int nonzeros,
                      const int* __restrict__ row_indices,
                      const float* __restrict__ values,
                      const int* __restrict__ row_offsets,
                      const int* __restrict__ column_indices,
                      const float* __restrict__ dense_matrix,
                      float* __restrict__ output_matrix, cudaStream_t stream,
                      int batch_size);
}