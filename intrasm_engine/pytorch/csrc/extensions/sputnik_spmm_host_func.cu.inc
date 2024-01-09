namespace my_sputnik {
namespace {

typedef std::function<cudaError_t(
    int,           // m: number of rows in lhs & output.
    int,           // k: number of cols in lhs and rows in rhs.
    int,           // n: number of cols in rhs/output.
    int,           // nonzeros: number of nonzero values in lhs.
    const int*,    // row_indices: ptr to row index swizzle map.
    const float*,  // values: ptr to lhs values.
    const int*,    // row_offsets: ptr to lhs row offsets.
    const int*,    // column_indices: ptr to lhs column indices.
    const float*,  // dense_matrix: ptr to rhs matrix.
    const float*,  // bias: bias pointer.
    float*,        // output_matrix: ptr to output matrix.
    cudaStream_t,
    int)>  // stream: stream to execute in.
    FloatSpmmFn2;

// Lookup table for kernel selection.
using FloatTable2 = std::unordered_map<std::string, FloatSpmmFn2>;

FloatTable2* GetFloatTable2() {
  static FloatTable2 kernel_table = {
      // MBV1 W1.8
      {MakeHandle(920, 920, 196, -1),
       CONCAT_ID(CudaSpmmEx, MYID) <
           SpmmConfig<float, float, float4, 1, 32, 128, 32> > },
      {MakeHandle(920, 464, 196, -1),
       CONCAT_ID(CudaSpmmEx, MYID) <
           SpmmConfig<float, float, float4, 1, 32, 128, 32> > },
      {MakeHandle(232, 115, 3136, -1),
       CONCAT_ID(CudaSpmmEx, MYID) <
           SpmmConfig<float, float, float4, 4, 8, 32, 8, 4, false> > },
      {MakeHandle(232, 232, 3136, -1),
       CONCAT_ID(CudaSpmmEx, MYID) <
           SpmmConfig<float, float2, float4, 4, 16, 32, 8, 4, false> > },
      // MBV1 W1.7
      {MakeHandle(872, 872, 196, -1),
       CONCAT_ID(CudaSpmmEx, MYID) <
           SpmmConfig<float, float, float4, 1, 32, 128, 32> > },
      {MakeHandle(872, 432, 196, -1),
       CONCAT_ID(CudaSpmmEx, MYID) <
           SpmmConfig<float, float, float4, 1, 32, 128, 32> > },
      {MakeHandle(216, 108, 3136, -1),
       CONCAT_ID(CudaSpmmEx, MYID) <
           SpmmConfig<float, float, float4, 4, 8, 32, 8, 4, false> > },
      {MakeHandle(216, 216, 3136, -1),
       CONCAT_ID(CudaSpmmEx, MYID) <
           SpmmConfig<float, float2, float4, 4, 16, 32, 8, 4, false> > },
      // MBV1 W1.6
      {MakeHandle(816, 816, 196, -1),
       CONCAT_ID(CudaSpmmEx, MYID) <
           SpmmConfig<float, float, float4, 1, 32, 128, 32> > },
      {MakeHandle(816, 408, 196, -1),
       CONCAT_ID(CudaSpmmEx, MYID) <
           SpmmConfig<float, float, float4, 1, 32, 128, 32> > },
      {MakeHandle(208, 102, 3136, -1),
       CONCAT_ID(CudaSpmmEx, MYID) <
           SpmmConfig<float, float, float4, 4, 8, 32, 8, 4, false> > },
      {MakeHandle(208, 208, 3136, -1),
       CONCAT_ID(CudaSpmmEx, MYID) <
           SpmmConfig<float, float2, float4, 4, 16, 32, 8, 4, false> > },
      // MBV1 W1.5
      {MakeHandle(768, 768, 196, -1),
       CONCAT_ID(CudaSpmmEx, MYID) <
           SpmmConfig<float, float, float4, 1, 32, 128, 32> > },
      {MakeHandle(768, 384, 196, -1),
       CONCAT_ID(CudaSpmmEx, MYID) <
           SpmmConfig<float, float, float4, 1, 32, 128, 32> > },
      {MakeHandle(192, 96, 3136, -1),
       CONCAT_ID(CudaSpmmEx, MYID) <
           SpmmConfig<float, float, float4, 4, 8, 32, 8, 4, false> > },
      {MakeHandle(192, 192, 3136, -1),
       CONCAT_ID(CudaSpmmEx, MYID) <
           SpmmConfig<float, float, float4, 4, 8, 32, 8, 4, false> > },
      // MBV1 W1.4
      {MakeHandle(720, 720, 196, -1),
       CONCAT_ID(CudaSpmmEx, MYID) <
           SpmmConfig<float, float, float4, 1, 32, 128, 32> > },
      {MakeHandle(720, 360, 196, -1),
       CONCAT_ID(CudaSpmmEx, MYID) <
           SpmmConfig<float, float, float4, 1, 32, 128, 32> > },
      {MakeHandle(176, 89, 3136, -1),
       CONCAT_ID(CudaSpmmEx, MYID) <
           SpmmConfig<float, float, float4, 4, 8, 32, 8, 4, false> > },
      {MakeHandle(176, 176, 3136, -1),
       CONCAT_ID(CudaSpmmEx, MYID) <
           SpmmConfig<float, float, float4, 4, 8, 32, 8, 4, false> > },
      // MBV1 W1.3
      {MakeHandle(664, 664, 196, -1),
       CONCAT_ID(CudaSpmmEx, MYID) <
           SpmmConfig<float, float, float4, 1, 32, 128, 32> > },
      {MakeHandle(664, 336, 196, -1),
       CONCAT_ID(CudaSpmmEx, MYID) <
           SpmmConfig<float, float, float4, 1, 32, 128, 32> > },
      {MakeHandle(168, 83, 3136, -1),
       CONCAT_ID(CudaSpmmEx, MYID) <
           SpmmConfig<float, float, float4, 4, 8, 32, 8, 4, false> > },
      {MakeHandle(168, 168, 3136, -1),
       CONCAT_ID(CudaSpmmEx, MYID) <
           SpmmConfig<float, float, float4, 4, 8, 32, 8, 4, false> > }};
  return &kernel_table;
}

FloatSpmmFn2 GetKernel(int m, int k, int n, int nonzeros) {
  FloatTable2* kernel_table = GetFloatTable2();
  auto it = kernel_table->find(MakeHandle(m, k, n, nonzeros));
  if (it == kernel_table->end()) {
    // Return uninitialized function to defer to the standard heuristic.
    FloatSpmmFn2 nullfn;
    return nullfn;
  }
  return it->second;
}
}  // namespace

cudaError_t CONCAT_ID(CudaSpmmBiasRelu, MYID)(
    int m, int k, int n, int nonzeros, const int* __restrict__ row_indices,
    const float* __restrict__ values, const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const float* __restrict__ dense_matrix, const float* __restrict__ bias,
    float* __restrict__ output_matrix, cudaStream_t stream, int batch_size) {
  // Try finding a specific kernel in the table. If we find a valid
  // one, call it and return.
  auto spmm_kernel = GetKernel(m, k, n, nonzeros);
  if (spmm_kernel) {
    return spmm_kernel(m, k, n, nonzeros, row_indices, values, row_offsets,
                       column_indices, dense_matrix, bias, output_matrix,
                       stream, batch_size);
  }

  // A very simple kernel selection heuristic. For small batch sizes,
  // we use the hybrid kernel variants with float4 sparse matrix loads.
  // For mid to large batch sizes, we use the standard float4 kernel with
  // and n-dimension tile of 32. On our synthetic RNN problem data this
  // gives us about 96% of the performance of a kernel selection oracle.
  //
  // TODO(tgale): We should improve the code here to make it more extensible
  // and less repetitive. We should also improve this heuristic to improve
  // performance on a wider range of problems.
  //
  // TODO(tgale): Update these heuristics to take batch size vector alignment
  // into account. This is currently not a perfectly general API.
  if ((n % 4) == 0) {
    if (n == 8) {
      // No predicates in the n-dimension.
      typedef SpmmConfig<float, float4, float, 4, 32, 8, 8, 4, false> Config;
      return CONCAT_ID(CudaSpmmEx, MYID)<Config>(
          m, k, n, nonzeros, row_indices, values, row_offsets, column_indices,
          dense_matrix, bias, output_matrix, stream, batch_size);
    } else if (n < 8) {
      typedef SpmmConfig<float, float4, float, 4, 32, 8, 8> Config;
      return CONCAT_ID(CudaSpmmEx, MYID)<Config>(
          m, k, n, nonzeros, row_indices, values, row_offsets, column_indices,
          dense_matrix, bias, output_matrix, stream, batch_size);
    } else if (n == 16) {
      // No predicates in the n-dimension.
      typedef SpmmConfig<float, float4, float2, 4, 32, 16, 8, 4, false> Config;
      return CONCAT_ID(CudaSpmmEx, MYID)<Config>(
          m, k, n, nonzeros, row_indices, values, row_offsets, column_indices,
          dense_matrix, bias, output_matrix, stream, batch_size);
    } else if (n < 16) {
      typedef SpmmConfig<float, float4, float2, 4, 32, 16, 8> Config;
      return CONCAT_ID(CudaSpmmEx, MYID)<Config>(
          m, k, n, nonzeros, row_indices, values, row_offsets, column_indices,
          dense_matrix, bias, output_matrix, stream, batch_size);
    } else if (n == 32) {
      // No predicates in the n-dimension.
      typedef SpmmConfig<float, float4, float4, 4, 32, 32, 8, 4, false> Config;
      return CONCAT_ID(CudaSpmmEx, MYID)<Config>(
          m, k, n, nonzeros, row_indices, values, row_offsets, column_indices,
          dense_matrix, bias, output_matrix, stream, batch_size);
    } else if ((n % 64) == 0) {
      // No predicates in n-dimension. Set kMinOccupancy to 8 to avoid
      // register spilling. Note that we only use this `large-tile` variant
      // if the batch size is divisble by 64.
      typedef SpmmConfig<float, float4, float4, 4, 32, 64, 8, 4, false, true, 8>
          Config;
      return CONCAT_ID(CudaSpmmEx, MYID)<Config>(
          m, k, n, nonzeros, row_indices, values, row_offsets, column_indices,
          dense_matrix, bias, output_matrix, stream, batch_size);
    } else {
      // Default kernel. 32-wide tile dimensions with 4-wide vector loads and
      // 4-way subwarp tiling. Run for all batch sizes greater than 16, unless
      // the batch size is divisible by 64.
      typedef SpmmConfig<float, float4, float4, 4, 32, 32, 8> Config;
      return CONCAT_ID(CudaSpmmEx, MYID)<Config>(
          m, k, n, nonzeros, row_indices, values, row_offsets, column_indices,
          dense_matrix, bias, output_matrix, stream, batch_size);
    }
  } else if ((n % 2) == 0) {
    typedef SpmmConfig<float, float2, float2, 2, 32, 32, 16> Config;
    return CONCAT_ID(CudaSpmmEx, MYID)<Config>(
        m, k, n, nonzeros, row_indices, values, row_offsets, column_indices,
        dense_matrix, bias, output_matrix, stream, batch_size);
  } else {
    // Scalar kernel.
    typedef SpmmConfig<float, float, float, 1, 32, 32, 32> Config;
    return CONCAT_ID(CudaSpmmEx, MYID)<Config>(
        m, k, n, nonzeros, row_indices, values, row_offsets, column_indices,
        dense_matrix, bias, output_matrix, stream, batch_size);
  }
}

cudaError_t CONCAT_ID(CudaSpmm, MYID)(int m, int k, int n, int nonzeros,
                                      const int* __restrict__ row_indices,
                                      const float* __restrict__ values,
                                      const int* __restrict__ row_offsets,
                                      const int* __restrict__ column_indices,
                                      const float* __restrict__ dense_matrix,
                                      float* __restrict__ output_matrix,
                                      cudaStream_t stream, int batch_size) {
  return CONCAT_ID(CudaSpmmBiasRelu, MYID)(
      m, k, n, nonzeros, row_indices, values, row_offsets, column_indices,
      dense_matrix,
      /* bias = */ nullptr, output_matrix, stream, batch_size);
}

}  // namespace my_sputnik
