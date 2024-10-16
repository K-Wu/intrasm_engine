from __future__ import annotations
from typing import Any
from time import sleep
import numpy as np
import torch
import torch.nn as nn
import intrasm_engine.common.triton_utils as triton_utils
import intrasm_engine.common.cutlass_utils as cutlass_utils
import intrasm_engine_extensions as iex
from intrasm_engine.pytorch.cpp_extensions.cuda_graph_constructor import (
    TorchCUDAGraphConstructor,
)
import cutlass
from cuda import cuda
from functools import partial
from triton_autotuning.matmul_lib import (
    MatmulTiling,
    MatrixLayout,
)
from intrasm_engine.common.compound_streams import CompoundStream
import contextlib

from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor

# custom interleaved linear layer class
# from intrasm_engine.pytorch.cpp_extensions.layers_and_funcs.temp import (
#     MyInterleavedLinear,
# )
from intrasm_engine.pytorch.cpp_extensions.layers_and_funcs.interleaved_linear_semistructured_half import (
    MyInterleavedModule,
)

# https://gist.github.com/mcarilli/376821aa1a7182dfcf59928a7cde3223
# --cuda-graph-trace graph (or node)
# https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
import nvtx

# reutrns dictionary of batched weights and inputs tensors
def generate_tensors(
    in_features, out_features, repeat_times
) -> dict[str, list[torch.Tensor]]:
    weights = [
        torch.randn(
            out_features, in_features, device="cuda", dtype=torch.float32
        )
        for _ in range(repeat_times)
    ]
    inputs = [
        torch.randn(1, in_features, device="cuda", dtype=torch.float32)
        for _ in range(repeat_times)
    ]

    return {"weights": weights, "inputs": inputs}

def generate_inputs(
    in_features, batch_size, repeat_times, type=torch.float32
) -> list[torch.Tensor]:
    if type == torch.float32:
        inputs = [
            torch.randn(
                batch_size, in_features, device = "cuda", dtype = torch.float32
            )
            for _ in range(repeat_times)
        ]
    else:
        inputs = [
            torch.randn(
                batch_size, in_features, device = "cuda", dtype = torch.float16
            )
            for _ in range(repeat_times)
        ]
    
    return inputs


def test_interleaved_linear(in_features, out_features, repeat_times):
    # interleaved_linear = MyInterleavedLinear()

    weights_and_inputs = generate_tensors(
        in_features, out_features, repeat_times
    )
    weights, inputs = (
        weights_and_inputs["weights"],
        weights_and_inputs["inputs"],
    )
    weights_cpu = [w.cpu() for w in weights]
    inputs_cpu = [i.cpu() for i in inputs]
    outputs = []

    print("weights shape: ", weights[0].shape)
    print("input shape: ", inputs[0].shape)

    constructor = TorchCUDAGraphConstructor()

    # time around this block
    # for i in range(repeat_times):
    out, out_sparse, out_dense = MyInterleavedLinear._forward(
        constructor, inputs[0], weights[0], constructor_enabled=True
    )
    outputs.append(out)
    # stop timer

    w_shape = weights_cpu[0].shape
    i_shape = inputs_cpu[0].shape

    out_cpu = out.cpu()
    out_golden = torch.matmul(weights_cpu[0], inputs_cpu[0].t())
    out_sparse_golden = torch.matmul(
        weights_cpu[0][:, 0 : w_shape[1] // 2],
        (inputs_cpu[0][:, 0 : i_shape[1] // 2]).t(),
    )
    out_dense_golden = torch.matmul(
        weights_cpu[0][:, w_shape[1] // 2 :],
        (inputs_cpu[0][:, i_shape[1] // 2 :]).t(),
    )
    # print("out sparse shape: ", out_sparse.shape)
    # print(out_sparse)
    # print("---------------------------------------------------------------")
    # print(out_sparse_golden)
    # print("out dense shape: ", out_dense.shape)
    # print(out_dense)
    # print("---------------------------------------------------------------")
    # print(out_dense_golden)

    # print(out_cpu.squeeze(0) == out_golden)
    # print(out_cpu.squeeze(0).t()) # transpose into 1 x out_features to make it easier to read
    # print(out_golden.t())

# @nvtx.annotate("interleaved", color = "blue")
def profile_interleaved_linear(layer, inputs_list):
    # move cuda events into layer for timing graph execution and concatentation
    # start_interleaved = torch.cuda.Event(enable_timing=True)
    # end_interleaved = torch.cuda.Event(enable_timing=True)
    
    # start_interleaved.record()
    for i in range(len(inputs_list)):
        # try:
        #     linear_interleaved(inputs_list[i])
        # except Exception as e:
        #     print(f"[ERROR]: {e}")
        linear_interleaved(inputs_list[i])
    # end_interleaved.record()
    # print(out.shape)
    # print(out_ld_test.shape)
    # # print(out_dense.shape)
    # print(out[0])
    # print(out[1])
    # print(out_ld_test[0])
    # print(out_ld_test[1])
    # # print(out_dense[0])
    torch.cuda.synchronize()
    # print(f"interleaved time for {repeat_times} times: ", start_interleaved.elapsed_time(end_interleaved))
    
@nvtx.annotate("normal linear", color = "red")
def profile_normal_linear(layer, inputs_list):
    start_golden = torch.cuda.Event(enable_timing=True)
    end_golden = torch.cuda.Event(enable_timing=True)
    
    start_golden.record()
    for i in range(repeat_times):
        out = linear_golden(inputs_list[i])
    end_golden.record()
    torch.cuda.synchronize()
    
    print(f"normal linear time for {repeat_times} times: ",start_golden.elapsed_time(end_golden))

def sleep_marker(sparsity, work_balance):
    @nvtx.annotate(f"sp:{sparsity}, wb:{work_balance}")
    def annotate():
        sleep(0.5)
    annotate()

if __name__ == "__main__":
    print(
        "testing interleaved linear layer"
    )
    # test_interleaved_linear(32, 64, 8)

    """ testing single inputs for correctness"""
    # mod = MyInterleavedModule(in_features=32, out_features=64)
    # input = torch.randn(4, 32, device="cuda", dtype=torch.float32)
    # weights = mod.weights
    # out = mod(input)
    # golden_dense = torch.matmul(input, weights[weights.shape[0] // 2 :, :].t())
    # golden_sparse = torch.matmul(
    #     input, weights[0 : weights.shape[0] // 2, :].t()
    # )
    # golden = torch.matmul(input, weights.t())
    # print(out)
    # print(golden)

    # input = torch.randn(4, 32, device="cuda", dtype=torch.float32)
    # out = mod(input)
    # golden = torch.matmul(input, weights.t())
    # print(out)
    # print(golden)
    
    """ testing repeated batched inputs for timing """

    # grid size determined by problem size / threadblock dimensions for m and n, along k dim, each block iterates thru all elems in k
    # NOTE: threadblock size might be referring to the size of the tile in terms of output matrix dimensions, not threads per dimensions,
    #       the warp_count determines how many threads are in each dimension. Based on input sizes (m,n,k), we tune the
    #       threadblock size to determine how many blocks we're launching, then try to fit the blocks onto the SMs to allow for interleaving
    #       based on the limitations of each SM (# of threads, # of warps, etc.)
    
    # problem right now is each block is eating too much shared mem (using 98k/block max is 99k), goal is to have each kernel use less shared mem, so we fit
    # more blocks/SM. Problem may be caused by tiles being too big or too little warps (each warp responsible for too much data)
    in_features = 768           # k
    out_features = 768      # n 
    batch_size = 4 * 1024       # m
    # in_features = 1024           # k
    # out_features = 1024      # n 
    # batch_size = 1024       # m
    repeat_times = 2
    
    # (m, k) * (k, n) = (m, n)
    # (128 * 41, 256) x (256, 256) = (128 * 41, 256)
    # weights are split by second dimension into (256, dense) and (256, sparse), sparse + dense = n
    
    sparsity_step = 0.2
    sparsity_range = np.arange(0.0, 1.0 + sparsity_step, sparsity_step)
    work_balance_step = 0.2
    work_balance_range = np.arange(work_balance_step, 1.0 + work_balance_step, work_balance_step)
    torch.set_printoptions(edgeitems=30)
    
    grid_search = False
    
    if grid_search:
        for sparsity in sparsity_range:
            for work_balance in work_balance_range:
                print(f"sparsity: {sparsity}")
                print(f"work_balance (fraction of work with GEMM): {work_balance}")
                sleep_marker(sparsity, work_balance)
                inputs_list = generate_inputs(in_features, batch_size, repeat_times)
                linear_interleaved = MyInterleavedModule(in_features=in_features, out_features=out_features, work_balance=work_balance)
                linear_interleaved.randomly_prune_weights(sparsity)
                
                linear_golden = torch.nn.Linear(in_features=in_features, out_features=out_features, device="cuda", dtype=torch.float32)
                # change linear_golden's weights to be the same as linear_interleaved
                with torch.no_grad():
                    linear_golden.weight = nn.Parameter(linear_interleaved.weights.clone().detach())
                
                
                profile_interleaved_linear(linear_interleaved, inputs_list)
                profile_normal_linear(linear_golden, inputs_list)
    else:
        inputs_list = generate_inputs(in_features, batch_size, repeat_times)
        inputs_list_fp16 = generate_inputs(in_features, batch_size, repeat_times, torch.float16)
        linear_interleaved = MyInterleavedModule(in_features=in_features, out_features=out_features)
        
        linear_interleaved.randomly_prune_weights(0.8)
        # linear_interleaved.prune_2_to_4()

        linear_golden = torch.nn.Linear(in_features=in_features, out_features=out_features, device="cuda", dtype=torch.float16)
        # change linear_golden's weights to be the same as linear_interleaved
        # with torch.no_grad():
        #     linear_golden.weight = nn.Parameter(linear_interleaved.weights.clone().detach())
    
        profile_interleaved_linear(linear_interleaved, inputs_list_fp16)
        profile_normal_linear(linear_golden, inputs_list_fp16)
    
    # nsys profile --cuda-graph-trace node -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -o test_interleaved_linear -f true -x true python ./temp.py
    