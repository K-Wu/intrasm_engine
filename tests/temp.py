from __future__ import annotations
from typing import Any
import torch
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

# custom interleaved linear layer class
from intrasm_engine.pytorch.cpp_extensions.layers_and_funcs.temp import (
    MyInterleavedLinear,
)
from intrasm_engine.pytorch.cpp_extensions.layers_and_funcs.fix import (
    MyInterleavedModule,
)


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


if __name__ == "__main__":
    print(
        "testing interleaved linear layer, non batched input, currently"
        " repeating ONLY ONCE"
    )
    # test_interleaved_linear(32, 64, 8)

    mod = MyInterleavedModule(in_features=32, out_features=64)
    input = torch.randn(2, 32, device="cuda", dtype=torch.float32)
    weights = mod.weights
    out = mod(input)
    golden_dense = torch.matmul(input, weights[weights.shape[0] // 2 :, :].t())
    golden_sparse = torch.matmul(
        input, weights[0 : weights.shape[0] // 2, :].t()
    )
    golden = torch.matmul(input, weights.t())
    # print(out)
    # print(golden)

    input = torch.randn(2, 32, device="cuda", dtype=torch.float32)
    out = mod(input)
    golden = torch.matmul(input, weights.t())
    # print(out)
    # print(golden)
