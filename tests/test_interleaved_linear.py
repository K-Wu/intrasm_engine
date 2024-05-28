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
from intrasm_engine.pytorch.cpp_extensions.layers_and_funcs.interleaved_linear import (
    MyInterleavedLinear,
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
    out = MyInterleavedLinear._forward(
        constructor, inputs[0], weights[0], constructor_enabled=True
    )
    outputs.append(out)
    # stop timer

    out_cpu = out.cpu()
    out_golden = torch.matmul(inputs_cpu[0], weights_cpu[0])
    # print(out_cpu == out_golden)
    print(out_cpu)
    print(out_golden)


if __name__ == "__main__":
    print(
        "testing interleaved linear layer, non batched input, currently"
        " repeating ONLY ONCE"
    )
    test_interleaved_linear(128, 128, 8)
