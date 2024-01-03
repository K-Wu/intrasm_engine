#include "../extensions.h"

at::Tensor spmm_sputnik_atomic(const at::Tensor& b,
                               const at::Tensor& row_indices,
                               const at::Tensor& values,
                               const at::Tensor& row_offsets,
                               const at::Tensor& column_indices, int64_t m) {
  TORCH_CHECK(b.dim() == 3);
  TORCH_CHECK(values.dim() == 2);
  TORCH_CHECK(b.size(0) == values.size(0));
  TORCH_CHECK(row_indices.dim() == 1);
  TORCH_CHECK(row_offsets.dim() == 1);
  TORCH_CHECK(column_indices.dim() == 1);
  TORCH_CHECK(values.size(1) == column_indices.size(0));

  TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
  TORCH_CHECK(row_indices.is_cuda(), "row_indices must be a CUDA tensor");
  TORCH_CHECK(values.is_cuda(), "values must be a CUDA tensor");
  TORCH_CHECK(row_offsets.is_cuda(), "row_offsets must be a CUDA tensor");
  TORCH_CHECK(column_indices.is_cuda(), "column_offsets must be a CUDA tensor");

  TORCH_CHECK(b.is_contiguous(), "b must be a contiguous tensor");
  TORCH_CHECK(row_indices.is_contiguous(),
              "row_indices must be a contiguous tensor");
  TORCH_CHECK(values.is_contiguous(), "values must be a contiguous tensor");
  TORCH_CHECK(row_offsets.is_contiguous(),
              "row_offsets must be a contiguous tensor");
  TORCH_CHECK(column_indices.is_contiguous(),
              "column_offsets must be a contiguous tensor");

  TORCH_CHECK(!b.is_sparse(), "b must be a dense tensor");
  TORCH_CHECK(!row_indices.is_sparse(), "row_indices must be a dense tensor");
  TORCH_CHECK(!values.is_sparse(), "values must be a dense tensor");
  TORCH_CHECK(!row_offsets.is_sparse(), "row_offsets must be a dense tensor");
  TORCH_CHECK(!column_indices.is_sparse(),
              "column_offsets must be a dense tensor");

  TORCH_CHECK(values.device() == b.device(),
              "values should be in the same device as b");
  TORCH_CHECK(values.device() == row_indices.device(),
              "a should be in the same device as row_indices");
  TORCH_CHECK(values.device() == row_offsets.device(),
              "a should be in the same device as row_offsets");
  TORCH_CHECK(values.device() == column_indices.device(),
              "a should be in the same device as column_indices");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int batch = b.size(0);
  int k = b.size(1);
  int n = b.size(2);

  int nonzeros = column_indices.size(0);
  TORCH_CHECK(
      batch == 1 || nonzeros % 4 == 0,
      "If batch size > 1 then number of nonzeros should be a multiple of 4");

  at::Tensor output = at::empty({batch, m, n}, b.options());

  // TODO investigate misaligned address errors in values ptr
  AT_CUDA_CHECK(my_sputnik::CudaSpmm2(
      m, k, n, nonzeros, row_indices.data_ptr<int>(), values.data_ptr<float>(),
      row_offsets.data_ptr<int>(), column_indices.data_ptr<int>(),
      b.data_ptr<float>(), output.data_ptr<float>(), stream, batch));

  return output;
}

TORCH_LIBRARY(iex_ops, m) {
  m.def("spmm_sputnik_atomic", &spmm_sputnik_atomic);
}