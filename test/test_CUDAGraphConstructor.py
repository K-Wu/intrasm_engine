import intrasm_engine
import intrasm_engine_extensions as iex  # Loading iex along without intrasm_engine will trigger libc10.so not found error
import torch
from intrasm_engine.pytorch.cpp_extensions.cuda_graph_constructor import (
    TorchCUDAGraphConstructor,
)

from intrasm_engine.common import cutlass_utils
from cuda import cuda
from typing import Callable

import intrasm_engine.common.triton_utils as triton_utils


def test_replay_torch(
    torch_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
):
    # intrasm_engine.pytorch.cpp_extensions.cuda_graph_constructor
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    c_ref = torch_func(a, b)
    constructor = TorchCUDAGraphConstructor()
    constructor.capture_library_call_begin()
    c = torch_func(a, b)
    constructor.capture_library_call_end()
    constructor.instantiate_graph_exec()
    constructor.execute_graph()
    constructor.synchronize()
    constructor.destroy_graph_exec()
    assert torch.allclose(c, c_ref)


def test_replay_torch_input_and_weight(
    torch_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
):
    # intrasm_engine.pytorch.cpp_extensions.cuda_graph_constructor
    a = torch.randn(512, 512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    c_ref = torch_func(a, b)
    constructor = TorchCUDAGraphConstructor()
    constructor.capture_library_call_begin()
    c = torch_func(a, b)
    constructor.capture_library_call_end()
    constructor.instantiate_graph_exec()
    constructor.execute_graph()
    constructor.synchronize()
    constructor.destroy_graph_exec()
    assert torch.allclose(c, c_ref)


def test_replay_torch_matmul():
    test_replay_torch(torch.matmul)


def test_replay_triton_matmul():
    test_replay_torch(triton_utils.run_matmul)


def test_replay_torch_matmul2():
    test_replay_torch_input_and_weight(torch.matmul)


def test_replay_torch_linear():
    test_replay_torch_input_and_weight(torch.nn.functional.linear)


def torch_linear_with_indexing(a: torch.Tensor, b: torch.Tensor):
    return torch.nn.functional.linear(a[2:514], b)


def test_replay_torch_linear_with_indexing():
    test_replay_torch(torch_linear_with_indexing)


def test_replay_cutlass_grouped_gemm():
    # This is based on https://github.com/NVIDIA/cutlass/blob/8236f30675bbe98f81d11c05764b77bfcb25b8cc/examples/python/02_pytorch_extension_grouped_gemm.ipynb.
    import cutlass

    dtype = torch.float16
    plan = cutlass.op.GroupedGemm(
        element=dtype, layout=cutlass.LayoutType.RowMajor
    )
    import random

    random.seed(2023)

    # Utility function to initialize A, B, C, and D matrices corresponding to dimensions M, N, and K
    def initialize(dtype, M, N, K):
        sizes = [(M, K), (K, N), (M, N), (M, N)]
        return [
            torch.randint(-3, 3, size, device="cuda").to(dtype)
            for size in sizes
        ]

    # Utility function to generate `problems` GEMMs of random sizes
    def generate_problems(problems):
        valid_sizes = [128, 256, 512, 1024]
        As, Bs, Cs, Ds = [], [], [], []
        for _ in range(problems):
            M, N, K = [random.choice(valid_sizes) for _ in range(3)]
            A, B, C, D = initialize(dtype, M, N, K)
            As.append(A)
            Bs.append(B)
            Cs.append(C)
            Ds.append(D)
        return As, Bs, Cs, Ds

    (
        As,
        Bs,
        Cs,
        Ds,
    ) = generate_problems(50)

    arguments = cutlass_utils.prepare_GemmGroupedArguments(
        plan, As, Bs, Cs, Ds, print_module=False
    )
    arguments.stream = cuda.CUstream(
        init_value=torch.cuda.current_stream().cuda_stream
    )
    constructor = TorchCUDAGraphConstructor()
    constructor.capture_library_call_begin()
    plan.operation.run(arguments)
    constructor.capture_library_call_end()
    constructor.instantiate_graph_exec()
    constructor.execute_graph()
    constructor.synchronize()
    constructor.destroy_graph_exec()
    Ds_torch = [a @ b for a, b in zip(As, Bs)]

    for d, d_torch in zip(Ds, Ds_torch):
        assert torch.allclose(d, d_torch)


def test_replay_cutlass_gemm():
    import cutlass

    dtype = torch.float16
    plan = cutlass.op.Gemm(
        element=dtype,
        layout=cutlass.LayoutType.RowMajor,
        element_C=cutlass.DataType.void,
        element_accumulator=cutlass.DataType.f16,
    )
    import random

    random.seed(2023)

    # Utility function to initialize A, B, C, and D matrices corresponding to dimensions M, N, and K
    def initialize(dtype, M, N, K):
        sizes = [(M, K), (K, N), (M, N)]
        return [
            torch.randint(-3, 3, size, device="cuda").to(dtype)
            for size in sizes
        ]

    # Utility function to generate `problems` GEMMs of random sizes
    def generate_problems():
        valid_sizes = [128, 256, 512, 1024]
        M, N, K = [random.choice(valid_sizes) for _ in range(3)]
        A, B, D = initialize(dtype, M, N, K)
        return A, B, D

    (
        A,
        B,
        D,
    ) = generate_problems()
    arguments = cutlass_utils.prepare_GemmArguments(
        plan,
        A,
        B,
        None,
        D,
        print_module=False,
        stream=cuda.CUstream(
            init_value=torch.cuda.current_stream().cuda_stream
        ),
    )
    constructor = TorchCUDAGraphConstructor()
    constructor.capture_library_call_begin()
    plan.operation.run(arguments)
    constructor.capture_library_call_end()
    constructor.instantiate_graph_exec()
    constructor.execute_graph()
    constructor.synchronize()
    constructor.destroy_graph_exec()

    D_torch = A @ B

    assert torch.allclose(D, D_torch)


if __name__ == "__main__":
    test_replay_torch_linear_with_indexing()
    test_replay_torch_linear()
    test_replay_torch_matmul()
    test_replay_torch_matmul2()
    test_replay_cutlass_grouped_gemm()
    test_replay_cutlass_gemm()
    test_replay_triton_matmul()
