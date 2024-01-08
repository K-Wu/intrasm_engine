import sys
import torch
from ...common import external_kernels  # sputnik and cutlass
from xformers.sparse.utils import _transpose_with_info

this = sys.modules[__name__]
this.single_kernel_autograd_functions: list[type[torch.autograd.Function]] = []

# In Sputnik SpMM implementation, A is sparse; B and C are row-major dense matrices.

# The forward and backward propagation of spmm/sddmm are following the scheme at https://github.com/facebookresearch/xformers/blob/6600003c2314af88befcec2cd6662957a662981d/xformers/sparse/_csr_ops.py
# Deducing the derivatives of SpMM/SDDMM: https://arxiv.org/abs/1909.01315


class MySpMM(torch.autograd.Function):
    """Based on _spmm at /xformers/sparse/_csr_ops.py"""

    # TODO: implement the csr/coo decision-making according to sparsity

    @staticmethod
    def forward(
        ctx,
        b,
        row_indices,
        values,
        row_offsets,
        column_indices,
        m,
        _transp_info,
        **kwargs
    ):
        constructor = kwargs["constructor"]
        backward_constructor = kwargs["backward_constructor"]
        num_streams = kwargs["num_streams"]
        assert num_streams == 1
        b = b.unsqueeze(0)
        b = (
            b.contiguous()
        )  # Set 1 as the "batch size" in sputnik's SpMM, i.e., set batch size as the "m" in sputnik's SpMM.
        # To use SpMM, set m as kBlockItemsY because each threadIdx.y works on only one row of A,.
        out = torch.ops.iex_ops.spmm_sputnik_reuse_weight(
            b, row_indices, values, row_offsets, column_indices, m
        )

        ctx.save_for_backward(
            b, row_indices, values, row_offsets, column_indices, *_transp_info
        )
        return out

    @staticmethod
    def backward(ctx, grad):
        num_streams = ctx.num_streams
        backward_constructor = ctx.backward_constructor
        (
            b,
            row_indices,
            values,
            row_offsets,
            column_indices,
            *_transp_info,
        ) = ctx.saved_tensors
        k = b.shape[1]

        # gradients w.r.t. values
        grad = grad.contiguous()
        grad = grad.unsqueeze(0)

        # To use SpMM, we need to transpose the dense input as well. (We are using the W*Dout(column-major) equivalence while the original is Dout(row-major)*W. )
        # Do the transpose out of the TorchCUDAConstructor (and timing) zone.
        # Do the reshape, etc. out of the TorchCUDAConstructor (and timing) zone as well.
        # set n as kBlockItemsX*kElementsPerScalar because each threadIdx.y works on only one row of A,.
        grad_sparse = torch.ops.iex_ops.sddmm_sputnik_atomic_upd_weight(
            grad, b, row_indices, row_offsets, column_indices
        )

        (
            row_indices_t,
            values_t,
            row_offsets_t,
            column_indices_t,
        ) = _transpose_with_info(values, _transp_info)

        grad_dense = torch.ops.iex_ops.spmm_sputnik_reuse_weight(
            grad, row_indices_t, values_t, row_offsets_t, column_indices_t, k
        )

        return grad_dense, None, grad_sparse, None, None, None, None
