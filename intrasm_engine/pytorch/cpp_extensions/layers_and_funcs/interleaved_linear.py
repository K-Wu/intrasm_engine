import sys
import torch
from cuda import cuda
from ....common import external_kernels  # sputnik and cutlass
from xformers.sparse.utils import _transpose_with_info
from xformers.components.attention.core import (
    SparseCS,
    _create_random_sparsity,
)
from intrasm_engine.pytorch.cpp_extensions.cuda_graph_constructor import (
    TorchCUDAGraphConstructor,
)
from .utils import MyAutogradFunc
from functools import partial

this = sys.modules[__name__]
this.single_kernel_autograd_functions: list[type[torch.autograd.Function]] = []

# In Sputnik SpMM implementation, A is sparse; B and C are row-major dense matrices.

# The forward and backward propagation of spmm/sddmm are following the scheme at https://github.com/facebookresearch/xformers/blob/6600003c2314af88befcec2cd6662957a662981d/xformers/sparse/_csr_ops.py
# Deducing the derivatives of SpMM/SDDMM: https://arxiv.org/abs/1909.01315


class MyInterleavedLinear(MyAutogradFunc):

    @staticmethod
    def num_inputs(**kwargs) -> int:
        return 3

    @staticmethod
    def num_input_tensors(**kwargs) -> int:
        return 3

    @staticmethod
    def num_saved_tensors(**ctx_kwargs) -> int:
        return 3


    @staticmethod
    def _forward(
        ctx,
        input,
        weight,
        **kwargs
    ):
        ctx.constructor = kwargs["constructor"]
        ctx.constructor.register_new_stream()
        
        # ctx.backward_constructor = kwargs["backward_constructor"]
        # ctx.num_streams = kwargs["num_streams"]
        ctx.constructor_enabled = kwargs["constructor_enabled"]
        # ctx.stream_beg = kwargs["stream_beg"]
        
        # weight is (out_features, in_features), split vertically (axis 1)
        # input is (1, in_features), split horizontally after transpose (axis 0)
        # operation weight * input^T
        w_shape = weight.shape
        i_shape = input.shape
        input = input.t()
        weight_sparse = weight[:, 0 : int(w_shape[1] / 2)]
        weight_dense = weight[:, int(w_shape[1] / 2) : ]
        in_sparse = input[0 : int(i_shape[0] / 2), :]
        in_dense = input[input(i_shape[0] / 2) , :].contiguous()
        
        out_dense = torch.randn(w_shape[0], 1, device = "cuda", dtype = torch.float32)
        out_sparse = torch.randn(w_shape[0], 1, device = "cuda", dtype = torch.float32)
        out = torch.randn(w_shape[0], 1, device = "cuda", dtype = torch.float32)
        
        # weight_csr = weight_sparse.to_sparse_csr()
        weight_csr = SparseCS(weight_sparse, torch.device("cuda"))
        # input kept as dense
                
        gemm_op = cutlass.op.Gemm(
            element = torch.float32,
            layout_A = cutlass.LayoutType.ColumnMajor,
            layout_B = cutlass.LayoutType.RowMajor,
            layout_C = cutlass.LayoutType.ColumnMajor,
            element_C = cutlass.DataType.void,
            element_accumulator = cutlass.DataType.f32,
        )
        
        gemm_args = cutlass.utils.prepare_GemmArguments(
            gemm_op,
            weight_dense,
            in_dense,
            None,
            out_dense,
            print_module = False,
            stream = cuda.CUstream(
                init_value=torch.cuda.current_stream().cuda_stream
            )
        )
        
        # b = b.contiguous()
        # Set 1 as the "batch size" in sputnik's SpMM, i.e., set batch size as the "m" in sputnik's SpMM.
        # b = b.unsqueeze(0)
        # To use SpMM, set m as kBlockItemsY because each threadIdx.y works on only one row of A.

        if (ctx.constructor_enabled):
            with ctx.constructor.registeredStreams[0] as compound_stream:
                
                ctx.constructor.capture_library_call_begin()
                # run dense gemm
                out_dense = gemm_op.operation.run(gemm_args)                
                ctx.constructor.capture_library_call_end()
                
            with ctx.constructor.registeredStreams[1] as compound_stream:
                ctx.constructor.capture_library_call_begin()
                
                out_sparse = torch.ops.iex_ops.spmm_sputnik_reuse_weight(
                    in_sparse,
                    weight_csr.row_indices,
                    weight_csr.values,
                    weight_csr.row_offsets,
                    weight_csr.column_indices,
                    w_shape[0]
                )
                
                ctx.constructor.capture_library_call_end()
                
            # accumulate results, wait for both streams to finish before executing
            out = out_dense + out_sparse
                
        else:
            out_dense = gemm_op.operation.run(gemm_args)
            out_sparse = torch.ops.iex_ops.spmm_sputnik_reuse_weight(
                in_sparse,
                weight_csr.row_indices,
                weight_csr.values,
                weight_csr.row_offsets,
                weight_csr.column_indices,
                w_shape[0]
            )
            out = out_dense + out_sparse
            
        # # is this graph already executed? like a one time graph creation
        # ctx.constructor.instantiate_graph_exec()
        # ctx.constructor.execute_graph()
        
        return out

    @staticmethod
    def _backward(ctx, grad):
        raise NotImplementedError("backward not implemented for interleaved linear yet")
    
