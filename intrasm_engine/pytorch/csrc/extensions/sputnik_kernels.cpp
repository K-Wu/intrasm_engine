#include "../extensions.h"

// TODO investigate misaligned address errors in values ptr
#define CONCAT_ID_(prefix, id) prefix##id
#define CONCAT_ID(prefix, id) CONCAT_ID_(prefix, id)

void check_values_with_weight_reuse_enabled(const at::Tensor& b,
                                            const at::Tensor& values,
                                            const at::Tensor& column_indices) {
  TORCH_CHECK(values.dim() == 1);
  TORCH_CHECK(values.size(0) == column_indices.size(0));
}

void check_values_with_weight_reuse_disabled(const at::Tensor& b,
                                             const at::Tensor& values,
                                             const at::Tensor& column_indices) {
  TORCH_CHECK(values.dim() == 2);
  TORCH_CHECK(b.size(0) == values.size(0));
  TORCH_CHECK(values.size(1) == column_indices.size(0));
}
at::Tensor empty_sddmm_output_with_weight_reuse_enabled(int batch, int nonzeros,
                                                        const at::Tensor& a) {
  return at::empty({nonzeros}, a.options());
}

at::Tensor empty_sddmm_output_with_weight_reuse_disabled(int batch,
                                                         int nonzeros,
                                                         const at::Tensor& a) {
  return at::empty({batch, nonzeros}, a.options());
}

#define DEFINE_SPMM_KERNEL_LAUNCHER(name, id, value_check_func, vtype_,        \
                                    itype_)                                    \
  at::Tensor name(const at::Tensor& b, const at::Tensor& row_indices,          \
                  const at::Tensor& values, const at::Tensor& row_offsets,     \
                  const at::Tensor& column_indices, int64_t m) {               \
    typedef vtype_ vtype;                                                      \
    typedef itype_ itype;                                                      \
    TORCH_CHECK(b.dim() == 3);                                                 \
    value_check_func(b, values, column_indices);                               \
    TORCH_CHECK(row_indices.dim() == 1);                                       \
    TORCH_CHECK(row_offsets.dim() == 1);                                       \
    TORCH_CHECK(column_indices.dim() == 1);                                    \
                                                                               \
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");                       \
    TORCH_CHECK(row_indices.is_cuda(), "row_indices must be a CUDA tensor");   \
    TORCH_CHECK(values.is_cuda(), "values must be a CUDA tensor");             \
    TORCH_CHECK(row_offsets.is_cuda(), "row_offsets must be a CUDA tensor");   \
    TORCH_CHECK(column_indices.is_cuda(),                                      \
                "column_offsets must be a CUDA tensor");                       \
                                                                               \
    TORCH_CHECK(b.is_contiguous(), "b must be a contiguous tensor");           \
    TORCH_CHECK(row_indices.is_contiguous(),                                   \
                "row_indices must be a contiguous tensor");                    \
    TORCH_CHECK(values.is_contiguous(), "values must be a contiguous tensor"); \
    TORCH_CHECK(row_offsets.is_contiguous(),                                   \
                "row_offsets must be a contiguous tensor");                    \
    TORCH_CHECK(column_indices.is_contiguous(),                                \
                "column_offsets must be a contiguous tensor");                 \
                                                                               \
    TORCH_CHECK(!b.is_sparse(), "b must be a dense tensor");                   \
    TORCH_CHECK(!row_indices.is_sparse(),                                      \
                "row_indices must be a dense tensor");                         \
    TORCH_CHECK(!values.is_sparse(), "values must be a dense tensor");         \
    TORCH_CHECK(!row_offsets.is_sparse(),                                      \
                "row_offsets must be a dense tensor");                         \
    TORCH_CHECK(!column_indices.is_sparse(),                                   \
                "column_offsets must be a dense tensor");                      \
                                                                               \
    TORCH_CHECK(values.device() == b.device(),                                 \
                "values should be in the same device as b");                   \
    TORCH_CHECK(values.device() == row_indices.device(),                       \
                "a should be in the same device as row_indices");              \
    TORCH_CHECK(values.device() == row_offsets.device(),                       \
                "a should be in the same device as row_offsets");              \
    TORCH_CHECK(values.device() == column_indices.device(),                    \
                "a should be in the same device as column_indices");           \
                                                                               \
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();                    \
    int batch = b.size(0);                                                     \
    int k = b.size(1);                                                         \
    int n = b.size(2);                                                         \
                                                                               \
    int nonzeros = column_indices.size(0);                                     \
    TORCH_CHECK(batch == 1 || nonzeros % 4 == 0,                               \
                "If batch size > 1 then number of nonzeros should be a "       \
                "multiple of 4");                                              \
                                                                               \
    at::Tensor output = at::empty({batch, m, n}, b.options());                 \
                                                                               \
    AT_CUDA_CHECK(my_sputnik::CONCAT_ID(CudaSpmm, id)(                         \
        m, k, n, nonzeros, row_indices.data_ptr<int>(),                        \
        values.data_ptr<vtype>(), row_offsets.data_ptr<int>(),                 \
        column_indices.data_ptr<itype>(), b.data_ptr<vtype>(),                 \
        output.data_ptr<vtype>(), stream, batch));                             \
                                                                               \
    return output;                                                             \
  }

#define DEFINE_SDDMM_KERNEL_LAUNCHER(name, id, empty_output_func, vtype_)      \
  at::Tensor name(                                                             \
      const at::Tensor& a, const at::Tensor& b, const at::Tensor& row_indices, \
      const at::Tensor& row_offsets, const at::Tensor& column_indices) {       \
    typedef vtype_ vtype;                                                      \
    TORCH_CHECK(a.dim() == b.dim());                                           \
    TORCH_CHECK(a.dim() == 3);                                                 \
    TORCH_CHECK(a.size(0) == b.size(0));                                       \
    TORCH_CHECK(a.size(2) == b.size(2));                                       \
    TORCH_CHECK(row_indices.dim() == 1);                                       \
    TORCH_CHECK(row_offsets.dim() == 1);                                       \
    TORCH_CHECK(column_indices.dim() == 1);                                    \
                                                                               \
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");                       \
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");                       \
    TORCH_CHECK(row_indices.is_cuda(), "row_indices must be a CUDA tensor");   \
    TORCH_CHECK(row_offsets.is_cuda(), "row_offsets must be a CUDA tensor");   \
    TORCH_CHECK(column_indices.is_cuda(),                                      \
                "column_offsets must be a CUDA tensor");                       \
                                                                               \
    TORCH_CHECK(a.is_contiguous(), "a must be a contiguous tensor");           \
    TORCH_CHECK(b.is_contiguous(), "b must be a contiguous tensor");           \
    TORCH_CHECK(row_indices.is_contiguous(),                                   \
                "row_indices must be a contiguous tensor");                    \
    TORCH_CHECK(row_offsets.is_contiguous(),                                   \
                "row_offsets must be a contiguous tensor");                    \
    TORCH_CHECK(column_indices.is_contiguous(),                                \
                "column_offsets must be a contiguous tensor");                 \
                                                                               \
    TORCH_CHECK(!a.is_sparse(), "a must be a dense tensor");                   \
    TORCH_CHECK(!b.is_sparse(), "b must be a dense tensor");                   \
    TORCH_CHECK(!row_indices.is_sparse(),                                      \
                "row_indices must be a dense tensor");                         \
    TORCH_CHECK(!row_offsets.is_sparse(),                                      \
                "row_offsets must be a dense tensor");                         \
    TORCH_CHECK(!column_indices.is_sparse(),                                   \
                "column_offsets must be a dense tensor");                      \
                                                                               \
    TORCH_CHECK(a.device() == b.device(),                                      \
                "a should be in the same device as b");                        \
    TORCH_CHECK(a.device() == row_indices.device(),                            \
                "a should be in the same device as row_indices");              \
    TORCH_CHECK(a.device() == row_offsets.device(),                            \
                "a should be in the same device as row_offsets");              \
    TORCH_CHECK(a.device() == column_indices.device(),                         \
                "a should be in the same device as column_indices");           \
                                                                               \
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();                    \
    int batch = a.size(0);                                                     \
    int m = a.size(1);                                                         \
    int k = a.size(2);                                                         \
    int n = b.size(1);                                                         \
                                                                               \
    int nonzeros = column_indices.size(0);                                     \
                                                                               \
    at::Tensor output = empty_output_func(batch, nonzeros, a);                 \
                                                                               \
    AT_CUDA_CHECK(my_sputnik::CONCAT_ID(CudaSddmm, id)(                        \
        m, k, n, nonzeros, row_indices.data_ptr<int>(),                        \
        row_offsets.data_ptr<int>(), column_indices.data_ptr<int>(),           \
        a.data_ptr<vtype>(), b.data_ptr<vtype>(), output.data_ptr<vtype>(),    \
        stream, batch));                                                       \
                                                                               \
    return output;                                                             \
  }

DEFINE_SPMM_KERNEL_LAUNCHER(spmm_sputnik_atomic, 2,
                            check_values_with_weight_reuse_disabled, float, int)
DEFINE_SPMM_KERNEL_LAUNCHER(spmm_sputnik_reuse_weight, 3,
                            check_values_with_weight_reuse_enabled, float, int)
// DEFINE_SPMM_KERNEL_LAUNCHER(spmm_sputnik_atomic, 4,
//                             check_values_with_weight_reuse_disabled,
//                             at::Half,short2)
// DEFINE_SPMM_KERNEL_LAUNCHER(spmm_sputnik_reuse_weight, 5,
//                             check_values_with_weight_reuse_enabled,
//                             at::Half,short2)
DEFINE_SDDMM_KERNEL_LAUNCHER(sddmm_sputnik_atomic_upd_weight, 2,
                             empty_sddmm_output_with_weight_reuse_enabled,
                             float)
// DEFINE_SDDMM_KERNEL_LAUNCHER(sddmm_sputnik_atomic_upd_weight, 3,
//                              empty_sddmm_output_with_weight_reuse_enabled,
//                              at::Half)

TORCH_LIBRARY(iex_ops, m) {
  m.def("spmm_sputnik_atomic", &spmm_sputnik_atomic);
  m.def("spmm_sputnik_reuse_weight", &spmm_sputnik_reuse_weight);
  m.def("sddmm_sputnik_atomic_upd_weight", &sddmm_sputnik_atomic_upd_weight);
}