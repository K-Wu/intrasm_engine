# nsys profile --cuda-graph-trace node -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -o test_gpt2 -f true -x true python ./test_gpt2.py

import sys
import torch
from cuda import cuda
from ....common import external_kernels  # sputnik and cutlass
import cutlass
# from xformers.sparse.utils import _transpose_with_info
# from xformers.components.attention.core import (
#     SparseCS,
# )
from ._sputnik_sparse import SparseCS
from intrasm_engine.pytorch.cpp_extensions.cuda_graph_constructor import (
    TorchCUDAGraphConstructor,
)
import intrasm_engine.common.cutlass_utils as cutlass_utils
from .utils import MyAutogradFunc
from functools import partial
import numpy as np
import nvtx

this = sys.modules[__name__]
this.single_kernel_autograd_functions: list[type[torch.autograd.Function]] = []

# In Sputnik SpMM implementation, A is sparse; B and C are row-major dense matrices.

# The forward and backward propagation of spmm/sddmm are following the scheme at https://github.com/facebookresearch/xformers/blob/6600003c2314af88befcec2cd6662957a662981d/xformers/sparse/_csr_ops.py
# Deducing the derivatives of SpMM/SDDMM: https://arxiv.org/abs/1909.01315

# DONE: generalize to batched input, allow stacked input tensor
# DONE: convert to pytorch standard matrix vector multiplication format (batch x input_features) * (input_features x out_features) = (batch x out_features)
#       check out torch.nn.Linear math for convention
# TODO: timing, use events class in torch.cuda submodule, create event right before and after execution on the main stream (stream[0]), check example
# DONE: nsight systems for timing, check timing overhead, nsight compute for occupancy of SMs
# TODO: sparseGPT: randomly prune non zeros in weights matrix (prune ~70% nonzeros), then replace linear layers. There's some docs that shows how to use it
# TODO: develop formula for determining correct tile descriptor given input shapes
# TODO: ratio of work between GEMM and sputnik, GEMM needs more work to balance execution times in graph
# TODO: check sputnik tiling information, we might need to grid search sputnik's tile descriptors
#       check sputnik_kernels.cpp, need new host function, same kernel (new function name, same ID),
#       new host file, reference sputnik_spmm_host_func.cu.inc, but need an input for which tiling config we use
# TODO: test cutlass tensorop using fp16 instead of float32, also test interleaved fp16 using both cutlass tensorop and sputnik
# NOTE: sputnik fp16 might end up scheduling work onto TC, so interleaving might not be very feasible
# TODO: comment out copying of inputs

# NOTE: check graph cublascusparse from playground for tiling
# NOTE: tiling in python with subscrription
# objective: configure the tests so that we can get each SM to have active blocks from both GEMM and SPMM kernels, this way we actually get the benefits from interleaving
#       1: customize tiling so that we can control the # of blocks from each kernel
#       2: customize input sizes


class MyInterleavedModule(torch.nn.Module):
    # put timing in module. self.time or something
    def __init__(self, in_features, out_features, bias=False):  # on GPU
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # TODO: write forula for determine dense_out_features and sparse_out_features
        self.dense_out_features = int(0.85 * out_features)
        # self.dense_out_features = out_features
        self.sparse_out_features = self.out_features - self.dense_out_features

        self.inputs = None
        self.in_sparse = None
        self.in_dense = None
        self.weights = torch.randn(
            out_features, in_features, device="cuda", dtype=torch.float32
        )
        self.bias = torch.randn(
            out_features, 1, device="cuda", dtype=torch.float32
        )
        # # split weights by first dimension (splitting out_features into 2)
        # self.weights_sparse_csr = SparseCS(
        #     self.weights[0 : out_features // 2, :], torch.device("cuda")
        # )  # keep as (out_features / 2, in_features), spmm does weights @ inputs^T
        # self.weights_dense = (
        #     self.weights[out_features // 2 :, :].t().contiguous()
        # )

        # using 192 for dense, 64 for sparse (based on the output size of 256, set by me in temp.py)
        self.weights_sparse_csr = SparseCS(
            self.weights[0 : self.sparse_out_features, :], torch.device("cuda")
        )  # keep as (out_features / 2, in_features), spmm does weights @ inputs^T
        self.weights_dense = (
            self.weights[self.sparse_out_features :, :].t().contiguous()
        )

        self.constructor = TorchCUDAGraphConstructor()

        # dimensions might change based on first input's batch size
        self.out = None
        self.out_dense = None
        self.out_sparse = None

    def randomly_prune_weights(self, proportion):
        rng = np.random.default_rng()
        self.weights = self.weights.cpu()
        with torch.no_grad():
            for i in range(self.out_features):
                print(f'loop :{i}')
                for j in range(self.in_features):
                    if rng.random() < proportion:
                        self.weights[i, j] = 0
        self.weights = self.weights.cuda()
        # split weights by first dimension (splitting out_features into 2)
        # self.weights_sparse_csr = SparseCS(
        #     self.weights[0 : self.out_features // 2, :], torch.device("cuda")
        # )  # keep as (out_features / 2, in_features), spmm does weights @ inputs^T
        # self.weights_dense = (
        #     self.weights[self.out_features // 2 :, :].t().contiguous()
        # )

        # using 192 for dense, 64 for sparse (based on the output size of 256, set by me in temp.py)
        with torch.no_grad():
            self.weights_sparse_csr = SparseCS(
                self.weights[0 : self.sparse_out_features, :], torch.device("cuda")
            )  # keep as (out_features / 2, in_features), spmm does weights @ inputs^T
            self.weights_dense = (
                self.weights[self.sparse_out_features :, :].t().contiguous()
            )

    def forward(self, x):  # copy new inputs into self.inputs
        size_out = x.size()[:-1] + (self.out_features,)
        if self.inputs == None:
            # this is the first input: set self.inputs and build graph
            # if x.dim() == 1:
            #     self.inputs = self.inputs.unsqueeze(0) # add a batch dimension
            #     self.out = torch.zeros(
            #         self.out_features, device='cuda', dtype=torch.float32
            #     )
            #     self.batch = False
            # else:
            #     self.out = torch.zeros(
            #         self.inputs.shape[0], self.out_features, device='cuda', dtype=torch.float32
            #     )
            #     self.batch = True
            self.copy_in = x
            self.inputs = self.copy_in.view(
                -1, x.size(-1)
            )  # (batch_features or 1, in_features)

            # TODO: x might be 1 dimensional, handle this case
            # self.out_dense = torch.zeros(
            #     self.inputs.shape[0],
            #     self.out_features // 2,
            #     device="cuda",
            #     dtype=torch.float32,
            # )
            # self.out_sparse = torch.zeros(
            #     self.inputs.shape[0],
            #     self.out_features // 2,
            #     device="cuda",
            #     dtype=torch.float32,
            # )

            # testing new division of work:
            self.out_dense = torch.zeros(
                self.inputs.shape[0],
                self.dense_out_features,
                device="cuda",
                dtype=torch.float32,
            )
            self.out_sparse = torch.zeros(
                self.inputs.shape[0],
                self.sparse_out_features,
                device="cuda",
                dtype=torch.float32,
            )

            # for this one, we don't split the inputs, we're only splitting weights, then concatenating results
            # self.in_sparse = (
            #     self.inputs.t().unsqueeze(0).contiguous()
            # )  # split into (batch, in_features / 2)
            # self.in_dense = self.inputs[:, self.in_features // 2 :]
            # print(self.in_sparse.shape)
            # print(self.weights_dense.shape)

            self.constructor.register_new_stream()
            gemm_op = cutlass.op.Gemm(
                element=torch.float32,
                layout=cutlass.LayoutType.RowMajor,
                element_C=cutlass.DataType.void,
                element_accumulator=cutlass.DataType.f32,
            )
            # tile_descr = gemm_op.tile_descriptions()
            # print(type(tile_descr))
            # for description in tile_descr:
            #     print(description.procedural_name())
            # gemm_op.opclass = cutlass.OpcodeClass.Simt
            # gemm_op.tile_description = {
            #     "threadblock_shape": [128, 64, 16],
            #     "warp_count": [4, 2, 1],
            #     "stages": 3
            # }

            """ verifying leading dimensions correctness """
            # self.out_ld_test = torch.zeros(
            #     self.inputs.shape[0],
            #     self.out_features,
            #     device="cuda",
            #     dtype=torch.float32,
            # )
            # gemm_ld_test = cutlass.op.Gemm(
            #     element=torch.float32,
            #     layout=cutlass.LayoutType.RowMajor,
            #     element_C=cutlass.DataType.void,
            #     element_accumulator=cutlass.DataType.f32,
            # )
            # gemm_ld_test.tile_description = {
            #     "threadblock_shape": [192, 64, 16],
            #     "warp_count": [4, 2, 1],
            #     "stages": 3,
            # }
            # gemm_ld_test_args = cutlass_utils.prepare_GemmArguments(
            #     gemm_ld_test,
            #     self.inputs,  # (batch, in_features), # (128 * 41, 256)
            #     self.weights_dense,  # (in_features, out_features / 2) # (256, 192) due to work distribution
            #     None,
            #     self.out_ld_test,  # (in_features, out_features), using self.out, but only filling out dense_out_features columns
            #     print_module=False,
            #     stream=self.constructor.registeredStreams[
            #         0
            #     ].torch_stream.stream.cuda_stream
            #     # problem_size_n=256
            # )
            # gemm_ld_test_args.ldc = self.out_features
            # gemm_ld_test_args.ldd = self.out_features # this gives a stride of out_features, which should skip the sparse part of output
            # gemm_ld_test_args.batched_stride_C = 128 * 41 * 256
            # gemm_ld_test_args.batched_stride_D = 128 * 41 * 256
            # print(gemm_ld_test_args.lda)
            # print(gemm_ld_test_args.ldb)
            # print(gemm_ld_test_args.ldc)
            # print(gemm_ld_test_args.ldd)
            # print(gemm_ld_test_args.batched_stride_C)
            # print(gemm_ld_test_args.batched_stride_D)
            """ end """

            gemm_op.tile_description = {
                "threadblock_shape": [128, 128, 16],
                "warp_count": [2, 2, 1],
                "stages": 3,
            }

            # prep arguments checks the metadata, but in the actual cutlass code, the inputs are passed via pointer, so we don't have to put this in the graph
            gemm_args = cutlass_utils.prepare_GemmArguments(
                gemm_op,
                self.inputs,  # (batch, in_features), # (128 * 41, 256)
                self.weights_dense,  # (in_features, out_features / 2) # (256, 192) due to work distribution
                None,
                self.out_dense,  # (in_features, dense_out_features), dense_out_features determined by work distribution
                print_module=False,
                stream=self.constructor.registeredStreams[
                    0
                ].torch_stream.stream.cuda_stream,
            )

            # print("lda: ", gemm_args.lda) # lda = in_features
            # print("ldb: ", gemm_args.ldb) # ldb = dense_out_features
            # print("ldc: ", gemm_args.ldc) # not using C
            # print("ldd: ", gemm_args.ldd) # ldd = dense_out_features

            # this is technically incorrect, but we put this here to get rid of the extra kernel in graph computation, for correctness, uncomment this call in stream 1
            self.in_sparse = (  # we have to put this in the graph since the transpose and contiguous may have memory copies going on
                self.inputs.t().unsqueeze(0).contiguous()
            )

            with self.constructor.registeredStreams[0] as compound_stream:
                # start interleeaved
                self.constructor.capture_library_call_begin()

                gemm_op.operation.run(gemm_args)
                # gemm_ld_test.operation.run(gemm_ld_test_args)

                self.constructor.capture_library_call_end()
            with self.constructor.registeredStreams[1] as compound_stream:
                self.constructor.capture_library_call_begin()

                # self.in_sparse = ( # we have to put this in the graph since the transpose and contiguous may have memory copies going on
                #     self.inputs.t().unsqueeze(0).contiguous()
                # )

                # self.out_sparse = (
                #     torch.ops.iex_ops.spmm_sputnik_reuse_weight(
                #         self.in_sparse,
                #         self.weights_sparse_csr.row_indices,
                #         self.weights_sparse_csr.values.squeeze(),
                #         self.weights_sparse_csr.row_offsets,
                #         self.weights_sparse_csr.column_indices,
                #         self.out_features // 2,
                #     )
                #     .squeeze(0)
                #     .t()
                # )
                self.out_sparse = (
                    torch.ops.iex_ops.spmm_sputnik_reuse_weight(
                        self.in_sparse,
                        self.weights_sparse_csr.row_indices,
                        self.weights_sparse_csr.values.squeeze(),
                        self.weights_sparse_csr.row_offsets,
                        self.weights_sparse_csr.column_indices,
                        self.sparse_out_features,
                    )
                    .squeeze(0)
                    .t()
                )

                self.constructor.capture_library_call_end()

            self.constructor.join(
                [self.constructor.registeredStreams[1]],
                self.constructor.registeredStreams[0],
            )
        elif (
            x.shape == self.copy_in.shape
        ):  # copy inputs and directly call apply on autograd function
            self.copy_in = self.copy_in.copy_(x)
        else:
            raise NotImplementedError("we assume input shapes are the same")

        MyInterleavedLinear._forward(
            self.constructor, constructor_enabled=True
        )

        with self.constructor.registeredStreams[0] as compound_stream:
            self.out = torch.concatenate(
                (self.out_sparse, self.out_dense), axis=1
            )

            # the following puts out_dense first, only used to conveniently check correctness when testing leading dims, revert back after testing
            # self.out = torch.concatenate(
            #     (self.out_dense, self.out_sparse), axis = 1
            # )

        # print("out_dense shape: ", self.out_dense.shape)
        # print("out_sparse shape: ", self.out_sparse.shape)

        # return self.out if self.batch else self.out.squeeze(), self.out_ld_test, self.out_dense
        # return self.out if self.batch else self.out.squeeze()
        # print(self.out.view(size_out).shape)
        return self.out.view(size_out)


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
    def _forward(constructor, **kwargs):
        constructor_enabled = kwargs["constructor_enabled"]
        
        start_interleaved = torch.cuda.Event(enable_timing=True)
        end_interleaved = torch.cuda.Event(enable_timing=True)

        @nvtx.annotate("interleaved_linear", color='blue')
        def exec():
            constructor.execute_graph()

        if constructor_enabled:
            constructor.instantiate_graph_exec()  # instantiates a graph execution object after we build the grpah object, CUDA graph API, measure this time, remove it from total

            # only time execute_graph()
            # constructor.execute_graph()  # actual execution of the graph
            start_interleaved.record()
            exec()
            end_interleaved.record()
            torch.cuda.synchronize()
            print(f"interleaved time: ", start_interleaved.elapsed_time(end_interleaved))

            constructor.destroy_graph_exec()
        else:
            raise NotImplementedError("we assume always using constructor")
        

    @staticmethod
    def _backward(ctx, grad):
        raise NotImplementedError(
            "backward not implemented for interleaved linear yet"
        )
