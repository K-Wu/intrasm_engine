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

from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor

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
# TODO: test cutlass tensorop using fp16 instead of float16, also test interleaved fp16 using both cutlass tensorop and sputnik
# NOTE: sputnik fp16 might end up scheduling work onto TC, so interleaving might not be very feasible
# TODO: comment out copying of inputs

# NOTE: check graph cublascusparse from playground for tiling
# NOTE: tiling in python with subscrription
# objective: configure the tests so that we can get each SM to have active blocks from both GEMM and SPMM kernels, this way we actually get the benefits from interleaving
#       1: customize tiling so that we can control the # of blocks from each kernel
#       2: customize input sizes

starts = []
ends = []

class MyInterleavedModule(torch.nn.Module):
    # put timing in module. self.time or something
    def __init__(self, in_features, out_features, bias=False):  # on GPU
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.inputs = None
        self.in_sparse = None
        self.in_dense = None
        self.weights = torch.randn(
            out_features, in_features, device="cuda", dtype=torch.float16
        ).t().contiguous()
        self.bias = torch.randn(
            out_features, 1, device="cuda", dtype=torch.float16
        )

        self.weights_sparse = torch.zeros_like(self.weights, dtype=torch.float16)
        self.weights_sparse_csr = SparseCS(
            self.weights_sparse, torch.device("cuda")
        ) 
        self.weights_dense = torch.zeros_like(self.weights, dtype=torch.float16)
        self.weights_semistructured = None

        self.constructor = TorchCUDAGraphConstructor()
        self.constructor.register_new_stream()

        # dimensions might change based on first input's batch size
        self.out = None
        self.out_dense = None
        self.out_sparse = None

    def randomly_prune_weights(self, proportion):
        rng = np.random.default_rng()
        # self.weights = self.weights.cpu()
        with torch.no_grad():
            for i in range(self.out_features):
                for j in range(0, self.in_features, 4):
                    non_zero_count = 0
                    for k in range(j, j + 4, 1):
                        rand = rng.random()
                        if rand >= proportion and non_zero_count > 2:
                            self.weights_sparse[i, k] = self.weights[i, k]
                        elif rand >= proportion and non_zero_count <= 2: 
                            self.weights_dense[i, k] = self.weights[i, k]
                            non_zero_count += 1
        # self.weights = self.weights.cuda()

            self.weights_sparse_csr = SparseCS(
                self.weights_sparse, torch.device("cuda")
            )
            self.weights_semistructured = to_sparse_semi_structured(self.weights_dense)

            
    def prune_2_to_4(self):
        self.weights = self.weights.cpu()
        pattern = [0, 0, 1, 1]
        with torch.no_grad():
            for i in range(self.out_features):
                for j in range(self.in_features):
                    self.weights[i, j] = pattern[j % 4] * self.weights[i, j]
        self.weights = self.weights.cuda()

    def forward(self, x):  # copy new inputs into self.inputs
        size_out = x.size()[:-1] + (self.out_features,)
        if self.inputs == None:
            # this is the first input: set self.inputs and build graph
            self.copy_in = x
            self.inputs = self.copy_in.view(
                -1, x.size(-1)
            )  # (batch_features or 1, in_features)

            # testing new division of work:
            self.out_dense = torch.zeros(
                self.inputs.shape[0],
                self.out_features,
                device="cuda",
                dtype=torch.float16,
            )
            self.out_sparse = torch.zeros(
                self.inputs.shape[0],
                self.out_features,
                device="cuda",
                dtype=torch.float16,
            )

            # this is technically incorrect, but we put this here to get rid of the extra kernel in graph execution, for correctness, uncomment this call in stream 1
            self.in_sparse = (  # we have to put this in the graph since the transpose and contiguous may have memory copies going on
                self.inputs.t().unsqueeze(0).contiguous()
            )
            self.inputs = self.inputs.t().contiguous()
                        
            start_gemm = torch.cuda.Event(enable_timing=True)
            start_gemm.record(
                stream=self.constructor.registeredStreams[0].torch_stream.stream
            )
            end_gemm = torch.cuda.Event(enable_timing=True)
            end_gemm.record(
                stream=self.constructor.registeredStreams[0].torch_stream.stream
            )
            
            start_spmm = torch.cuda.Event(enable_timing=True)
            start_spmm.record(
                stream=self.constructor.registeredStreams[1].torch_stream.stream
            )
            end_spmm = torch.cuda.Event(enable_timing=True)
            end_spmm.record(
                stream=self.constructor.registeredStreams[1].torch_stream.stream
            )
            
            starts.append(start_gemm)
            starts.append(start_spmm)
            ends.append(end_gemm)
            ends.append(end_spmm)
            
            with self.constructor.registeredStreams[0] as compound_stream:
                self.constructor.add_event_record_node(
                    start_gemm,
                    self.constructor.registeredStreams[0].torch_stream
                )
                self.constructor.capture_library_call_begin()

                self.out_dense = torch.mm(
                    self.weights_semistructured, self.inputs
                )

                self.constructor.capture_library_call_end()
                self.constructor.add_event_record_node(
                    end_gemm,
                    self.constructor.registeredStreams[0].torch_stream
                )
            with self.constructor.registeredStreams[1] as compound_stream:
                self.constructor.add_event_record_node(
                    start_spmm,
                    self.constructor.registeredStreams[1].torch_stream
                )
                self.constructor.capture_library_call_begin()
                
                # pass compound_stream's torch stream into a timing enabled cuda event
                # self.in_sparse = ( # we have to put this in the graph since the transpose and contiguous may have memory copies going on
                #     self.inputs.t().unsqueeze(0).contiguous()
                # )
                self.out_sparse = (
                    torch.ops.iex_ops.spmm_sputnik_reuse_weight_half(
                        self.in_sparse,
                        self.weights_sparse_csr.row_indices,
                        # self.weights_sparse_csr.row_indices.to(torch.int16),
                        self.weights_sparse_csr.values.squeeze(),
                        self.weights_sparse_csr.row_offsets,
                        # self.weights_sparse_csr.row_offsets.to(torch.int16),
                        # self.weights_sparse_csr.column_indices,
                        self.weights_sparse_csr.column_indices.to(torch.int16),
                        self.out_features,
                    )
                    # .squeeze(0)
                    # .t() # moved this part to after the graph exec
                )

                self.constructor.capture_library_call_end()
                self.constructor.add_event_record_node(
                    end_spmm,
                    self.constructor.registeredStreams[1].torch_stream
                )
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
            self.out = self.out_sparse.squeeze(0).t() + self.out_dense.t().contiguous()
            
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

        if constructor_enabled:
            constructor.instantiate_graph_exec()  # instantiates a graph execution object after we build the grpah object, CUDA graph API, measure this time, remove it from total

            with nvtx.annotate('interleaved_linear', color='blue'):
                constructor.execute_graph()
            torch.cuda.synchronize()
                
            print('gemm_time: ', starts[0].elapsed_time(ends[0]))
            print('spmm_time: ', starts[1].elapsed_time(ends[1]))
            print('overlap_time: ', max(
                starts[0].elapsed_time(ends[0]),
                starts[0].elapsed_time(ends[1]),
                starts[1].elapsed_time(ends[0]),
                starts[1].elapsed_time(ends[1])
            ))

            constructor.destroy_graph_exec()
        else:
            raise NotImplementedError("we assume always using constructor")
        

    @staticmethod
    def _backward(ctx, grad):
        raise NotImplementedError(
            "backward not implemented for interleaved linear yet"
        )
