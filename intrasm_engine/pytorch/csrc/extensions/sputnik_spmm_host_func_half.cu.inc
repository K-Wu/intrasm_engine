namespace my_sputnik {

cudaError_t CONCAT_ID(CudaSpmmBiasRelu, MYID)(
    int m, int k, int n, int nonzeros, const int* __restrict__ row_indices,
    const half2* __restrict__ values, const int* __restrict__ row_offsets,
    const short2* __restrict__ column_indices,
    const half2* __restrict__ dense_matrix, const float* __restrict__ bias,
    half2* __restrict__ output_matrix, cudaStream_t stream) {
  // Simple kernel selction heuristic for half-precision kernels. For batch
  // sizes of 16 or less we use hybrid variants with half8 sparse matrix
  // loads and half2 dense matrix loads/stores. For batch size 32 or less we
  // use the hybrid variant with half8/half4 memory ops. For larger batch
  // sizes, we use the half4 variants, since half8 variants run into register
  // issues with predication enabled. If the batch size is divisbile by one
  // of our tile sizes, we disable predicates and use the full half8 kernels.
  //
  // TODO(tgale): Look into whether setting our launch bounds lets us avoid
  // spilling on some of the larger tile variants.
  if (n < 16) {
    typedef SpmmConfig<half2, half8, half2, 4, 32, 8, 8, 4> Config;
    return CONCAT_ID(CudaSpmmEx, MYID)<Config>(
        m, k, n, nonzeros, row_indices, values, row_offsets, column_indices,
        dense_matrix, bias, output_matrix, stream);
  } else if (n == 16) {
    typedef SpmmConfig<half2, half8, half2, 4, 32, 8, 8, 4, false> Config;
    return CONCAT_ID(CudaSpmmEx, MYID)<Config>(
        m, k, n, nonzeros, row_indices, values, row_offsets, column_indices,
        dense_matrix, bias, output_matrix, stream);
  } else if (n < 32) {
    typedef SpmmConfig<half2, half8, half4, 4, 32, 16, 8, 4> Config;
    return CONCAT_ID(CudaSpmmEx, MYID)<Config>(
        m, k, n, nonzeros, row_indices, values, row_offsets, column_indices,
        dense_matrix, bias, output_matrix, stream);
  } else if (n == 32) {
    typedef SpmmConfig<half2, half8, half4, 4, 32, 16, 8, 4, false> Config;
    return CONCAT_ID(CudaSpmmEx, MYID)<Config>(
        m, k, n, nonzeros, row_indices, values, row_offsets, column_indices,
        dense_matrix, bias, output_matrix, stream);
  } else if (n > 32 && ((n % 64) == 0)) {
    typedef SpmmConfig<half2, half8, half8, 4, 32, 32, 8, 4, false> Config;
    return CONCAT_ID(CudaSpmmEx, MYID)<Config>(
        m, k, n, nonzeros, row_indices, values, row_offsets, column_indices,
        dense_matrix, bias, output_matrix, stream);
  } else {
    typedef SpmmConfig<half2, half4, half4, 2, 32, 32, 16, 4> Config;
    return CONCAT_ID(CudaSpmmEx, MYID)<Config>(
        m, k, n, nonzeros, row_indices, values, row_offsets, column_indices,
        dense_matrix, bias, output_matrix, stream);
  }
}

cudaError_t CONCAT_ID(CudaSpmm, MYID)(int m, int k, int n, int nonzeros,
                                      const int* __restrict__ row_indices,
                                      const half2* __restrict__ values,
                                      const int* __restrict__ row_offsets,
                                      const short2* __restrict__ column_indices,
                                      const half2* __restrict__ dense_matrix,
                                      half2* __restrict__ output_matrix,
                                      cudaStream_t stream) {
  return CONCAT_ID(CudaSpmmBiasRelu, MYID)(
      m, k, n, nonzeros, row_indices, values, row_offsets, column_indices,
      dense_matrix, /* bias = */ nullptr, output_matrix, stream);
}

}  // namespace my_sputnik