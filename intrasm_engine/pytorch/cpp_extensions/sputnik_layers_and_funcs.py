import sys
import torch
from ...common import external_kernels  # sputnik and cutlass
from xformers.sparse.utils import _transpose_with_info
from intrasm_engine.pytorch.cpp_extensions.cuda_graph_constructor import (
    TorchCUDAGraphConstructor,
)
from .layers_and_funcs_utils import MyAutogradFunc

this = sys.modules[__name__]
this.single_kernel_autograd_functions: list[type[torch.autograd.Function]] = []

# In Sputnik SpMM implementation, A is sparse; B and C are row-major dense matrices.

# The forward and backward propagation of spmm/sddmm are following the scheme at https://github.com/facebookresearch/xformers/blob/6600003c2314af88befcec2cd6662957a662981d/xformers/sparse/_csr_ops.py
# Deducing the derivatives of SpMM/SDDMM: https://arxiv.org/abs/1909.01315


class MySpMM(MyAutogradFunc):
    """Based on _spmm at /xformers/sparse/_csr_ops.py"""

    @staticmethod
    def num_inputs(**kwargs) -> int:
        return 3

    @staticmethod
    def num_input_tensors(**kwargs) -> int:
        return 3

    @staticmethod
    def num_saved_tensors(**ctx_kwargs) -> int:
        return 3

    # TODO: implement the csr/coo decision-making according to sparsity

    @staticmethod
    def _forward(
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
        ctx.constructor = kwargs["constructor"]
        ctx.backward_constructor = kwargs["backward_constructor"]
        ctx.num_streams = kwargs["num_streams"]
        ctx.constructor_enabled = kwargs["constructor_enabled"]
        ctx.stream_beg = kwargs["stream_beg"]
        assert ctx.num_streams == 1
        b = b.contiguous()
        # Set 1 as the "batch size" in sputnik's SpMM, i.e., set batch size as the "m" in sputnik's SpMM.
        b = b.unsqueeze(0)
        # To use SpMM, set m as kBlockItemsY because each threadIdx.y works on only one row of A.
        if ctx.constructor_enabled:
            with ctx.constructor.registeredStreams[
                ctx.stream_beg
            ] as compund_stream:
                ctx.constructor.capture_library_call_begin()
                out = torch.ops.iex_ops.spmm_sputnik_reuse_weight(
                    b, row_indices, values, row_offsets, column_indices, m
                )
                ctx.constructor.capture_library_call_end()
        else:
            out = torch.ops.iex_ops.spmm_sputnik_reuse_weight(
                b, row_indices, values, row_offsets, column_indices, m
            )

        ctx.save_for_backward(
            b, row_indices, values, row_offsets, column_indices, *_transp_info
        )
        return out

    @staticmethod
    def _backward(ctx, grad):
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
        if ctx.need_input_grad[2]:
            if ctx.constructor_enabled:
                with ctx.backward_constructor.registeredStreams[
                    ctx.stream_beg
                ] as compund_stream:
                    ctx.backward_constructor.capture_library_call_begin()
                    grad_sparse = (
                        torch.ops.iex_ops.sddmm_sputnik_atomic_upd_weight(
                            grad, b, row_indices, row_offsets, column_indices
                        )
                    )
                    ctx.backward_constructor.capture_library_call_end()
            else:
                grad_sparse = (
                    torch.ops.iex_ops.sddmm_sputnik_atomic_upd_weight(
                        grad, b, row_indices, row_offsets, column_indices
                    )
                )

        if ctx.need_input_grad[0]:
            (
                row_indices_t,
                values_t,
                row_offsets_t,
                column_indices_t,
            ) = _transpose_with_info(values, _transp_info)

            if ctx.constructor_enabled:
                with ctx.backward_constructor.registeredStreams[
                    ctx.stream_beg
                ] as compund_stream:
                    ctx.backward_constructor.capture_library_call_begin()
                    grad_dense = torch.ops.iex_ops.spmm_sputnik_reuse_weight(
                        grad,
                        row_indices_t,
                        values_t,
                        row_offsets_t,
                        column_indices_t,
                        k,
                    )
                    ctx.backward_constructor.capture_library_call_end()
            else:
                grad_dense = torch.ops.iex_ops.spmm_sputnik_reuse_weight(
                    grad,
                    row_indices_t,
                    values_t,
                    row_offsets_t,
                    column_indices_t,
                    k,
                )
            grad_dense = grad_dense.squeeze(0)

        return grad_dense, None, grad_sparse, None, None, None, None


class MySpMMPartitioned(MyAutogradFunc):
    ...
    # TODO: implement this
