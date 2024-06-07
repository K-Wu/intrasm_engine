import sys
import torch
from cuda import cuda
from ....common import external_kernels  # sputnik and cutlass
import cutlass
from xformers.sparse.utils import _transpose_with_info
from xformers.components.attention.core import (
    SparseCS,
    _create_random_sparsity,
)
from intrasm_engine.pytorch.cpp_extensions.cuda_graph_constructor import (
    TorchCUDAGraphConstructor,
)
import intrasm_engine.common.cutlass_utils as cutlass_utils
from .utils import MyAutogradFunc
from functools import partial

this = sys.modules[__name__]
this.single_kernel_autograd_functions: list[type[torch.autograd.Function]] = []

# In Sputnik SpMM implementation, A is sparse; B and C are row-major dense matrices.

# The forward and backward propagation of spmm/sddmm are following the scheme at https://github.com/facebookresearch/xformers/blob/6600003c2314af88befcec2cd6662957a662981d/xformers/sparse/_csr_ops.py
# Deducing the derivatives of SpMM/SDDMM: https://arxiv.org/abs/1909.01315

# TODO: generalize to batched input, allow stacked input tensor
# TODO: convert to pytorch standard matrix vector multiplication format (batch x input_features) * (input_features x out_features) = (batch x out_features)
#       check out torch.nn.Linear math for convention
# TODO: timing, use events class in torch.cuda submodule, create event right before and after execution on the main stream (stream[0]), check example
# TODO: sparseGPT: randomly prune non zeros in weights matrix (prune ~70% nonzeros), then replace linear layers. There's some docs that shows how to use it


class MyInterleavedModule(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False):  # on GPU
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.inputs = None
        self.in_sparse = None
        self.in_dense = None
        self.weights = torch.randn(
            out_features, in_features, device="cuda", dtype=torch.float32
        )
        self.bias = torch.randn(
            out_features, 1, device="cuda", dtype=torch.float32
        )
        # split weights by first dimension (splitting out_features into 2)
        self.weights_sparse_csr = SparseCS(
            self.weights[0 : out_features // 2, :], torch.device("cuda")
        )  # keep as (out_features / 2, in_features), spmm does weights @ inputs^T
        self.weights_dense = (
            self.weights[out_features // 2 :, :].t().contiguous()
        )

        self.constructor = TorchCUDAGraphConstructor()

        # dimensions might change based on first input's batch size
        self.out_dense = None
        self.out_sparse = None

    def forward(self, x):  # copy new inputs into self.inputs
        if self.inputs == None:
            # this is the first input: set self.inputs and build graph
            self.inputs = x  # x is (batch, in_features)

            # TODO: x might be 1 dimensional, handle this case
            self.out_dense = torch.zeros(
                x.shape[0],
                self.out_features // 2,
                device="cuda",
                dtype=torch.float32,
            )
            self.out_sparse = torch.zeros(
                x.shape[0],
                self.out_features // 2,
                device="cuda",
                dtype=torch.float32,
            )

            # for this one, we don't split the inputs, we're only splitting weights, then concatenating results
            self.in_sparse = (
                self.inputs.t().unsqueeze(0).contiguous()
            )  # split into (batch, in_features / 2)
            # self.in_dense = self.inputs[:, self.in_features // 2 :]
            print(self.in_sparse.shape)
            # print(self.weights_dense.shape)

            self.constructor.register_new_stream()
            gemm_op = cutlass.op.Gemm(
                element=torch.float32,
                layout=cutlass.LayoutType.RowMajor,
                element_C=cutlass.DataType.void,
                element_accumulator=cutlass.DataType.f32,
            )
            gemm_args = cutlass_utils.prepare_GemmArguments(
                gemm_op,
                self.inputs,  # (batch, in_features)
                self.weights_dense,  # (in_features, out_features / 2)
                None,
                self.out_dense,
                print_module=False,
                stream=self.constructor.registeredStreams[
                    0
                ].torch_stream.stream.cuda_stream,
            )

            with self.constructor.registeredStreams[0] as compound_stream:
                self.constructor.capture_library_call_begin()
                # run dense gemm
                gemm_op.operation.run(gemm_args)
                self.constructor.capture_library_call_end()
            with self.constructor.registeredStreams[1] as compound_stream:
                self.constructor.capture_library_call_begin()

                self.out_sparse = (
                    torch.ops.iex_ops.spmm_sputnik_reuse_weight(
                        self.in_sparse,
                        self.weights_sparse_csr.row_indices,
                        self.weights_sparse_csr.values.squeeze(),
                        self.weights_sparse_csr.row_offsets,
                        self.weights_sparse_csr.column_indices,
                        self.out_features // 2,
                    )
                    .squeeze(0)
                    .t()
                )

                self.constructor.capture_library_call_end()

            self.constructor.join(
                [self.constructor.registeredStreams[1]],
                self.constructor.registeredStreams[0],
            )
        else:  # copy inputs and directly call apply on autograd function
            self.inputs = self.inputs.copy_(x)

        MyInterleavedLinearNew._forward(
            self.constructor, constructor_enabled=True
        )

        # return self.out_sparse.squeeze(0).t(), self.out_dense

        with self.constructor.registeredStreams[0] as compound_stream:
            self.out = torch.concatenate(
                (self.out_sparse, self.out_dense), axis=1
            )

        return self.out


class MyInterleavedLinearNew(MyAutogradFunc):
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
    def _forward(constructor, **kwargs):  # single threaded operation
        constructor_enabled = kwargs["constructor_enabled"]

        if constructor_enabled:
            constructor.instantiate_graph_exec()  # instantiates a graph execution object after we build the grpah object, CUDA graph API, measure this time, remove it from total
            constructor.execute_graph()  # actual execution of the graph
            constructor.destroy_graph_exec()
        else:
            raise NotImplementedError("we assume always using constructor")

    @staticmethod
    def _backward(ctx, grad):
        raise NotImplementedError(
            "backward not implemented for interleaved linear yet"
        )


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
    def _forward(  # single threaded operation
        constructor, input, weight, **kwargs
    ):
        # constructor = kwargs["constructor"]

        if len(constructor.registeredStreams) < 2:
            constructor.register_new_stream()  # perhaps check if there are already 2 streams there

        # ctx.backward_constructor = kwargs["backward_constructor"]
        # ctx.num_streams = kwargs["num_streams"]
        constructor_enabled = kwargs["constructor_enabled"]
        # ctx.stream_beg = kwargs["stream_beg"]

        # weight is (out_features, in_features), split vertically (axis 1)
        # input is (1, in_features), split horizontally after transpose (axis 0)
        # operation weight * input^T
        w_shape = weight.shape

        input = input.t()
        i_shape = input.shape
        weight_sparse = weight[:, 0 : w_shape[1] // 2]
        weight_dense = weight[:, w_shape[1] // 2 :].contiguous()
        in_sparse = (input[0 : i_shape[0] // 2, :]).unsqueeze(0)
        in_dense = input[i_shape[0] // 2 :, :].contiguous()

        out_dense = torch.randn(
            w_shape[0], 1, device="cuda", dtype=torch.float32
        )
        out_sparse = torch.randn(
            w_shape[0], 1, device="cuda", dtype=torch.float32
        )
        out = torch.zeros(w_shape[0], 1, device="cuda", dtype=torch.float32)

        # weight_csr = weight_sparse.to_sparse_csr()
        weight_csr = SparseCS(weight_sparse, torch.device("cuda"))
        # input kept as dense

        gemm_op = cutlass.op.Gemm(
            element=torch.float32,
            layout=cutlass.LayoutType.RowMajor,
            # layout_A = cutlass.LayoutType.ColumnMajor,
            # layout_B = cutlass.LayoutType.RowMajor,
            # layout_C = cutlass.LayoutType.ColumnMajor,
            element_C=cutlass.DataType.void,
            element_accumulator=cutlass.DataType.f32,
        )

        # gemm_args = cutlass_utils.prepare_GemmArguments(
        #     gemm_op,
        #     weight_dense,
        #     in_dense,
        #     None,
        #     out_dense,
        #     print_module = False,
        #     stream = constructor.registeredStreams[0].torch_stream.stream.cuda_stream
        # )
        gemm_args = cutlass_utils.prepare_GemmArguments(
            gemm_op,
            weight_dense,
            in_dense,
            None,
            out_dense,
            print_module=False,
            stream=constructor.registeredStreams[
                0
            ].torch_stream.stream.cuda_stream,
        )

        # b = b.contiguous()
        # Set 1 as the "batch size" in sputnik's SpMM, i.e., set batch size as the "m" in sputnik's SpMM.
        # b = b.unsqueeze(0)
        # To use SpMM, set m as kBlockItemsY because each threadIdx.y works on only one row of A.

        if constructor_enabled:
            with constructor.registeredStreams[0] as compound_stream:
                constructor.capture_library_call_begin()
                # run dense gemm
                gemm_op.operation.run(gemm_args)
                constructor.capture_library_call_end()

            with constructor.registeredStreams[1] as compound_stream:
                constructor.capture_library_call_begin()

                out_sparse = torch.ops.iex_ops.spmm_sputnik_reuse_weight(
                    in_sparse,
                    weight_csr.row_indices,
                    weight_csr.values.squeeze(),
                    weight_csr.row_offsets,
                    weight_csr.column_indices,
                    w_shape[0],
                )

                constructor.capture_library_call_end()

            constructor.join(
                [constructor.registeredStreams[1]],
                constructor.registeredStreams[0],
            )

            # accumulate results, wait for both streams to finish before executing
            constructor.instantiate_graph_exec()  # instantiates a graph execution object after we build the grpah object, CUDA graph API, measure this time, remove it from total
            constructor.execute_graph()  # actual execution of the graph

            with constructor.registeredStreams[0] as compound_stream:
                out = out_dense + out_sparse

        else:
            out_dense = gemm_op.operation.run(gemm_args)
            out_sparse = torch.ops.iex_ops.spmm_sputnik_reuse_weight(
                in_sparse,
                weight_csr.row_indices,
                weight_csr.values,
                weight_csr.row_offsets,
                weight_csr.column_indices,
                w_shape[0],
            )
            out = out_dense + out_sparse

        # # is this graph already executed? like a one time graph creation
        # constructor.instantiate_graph_exec()
        # constructor.execute_graph()

        return out, out_sparse, out_dense

    @staticmethod
    def _backward(ctx, grad):
        raise NotImplementedError(
            "backward not implemented for interleaved linear yet"
        )
