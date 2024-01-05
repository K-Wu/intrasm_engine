import intrasm_engine
import intrasm_engine_extensions as iex  # Loading iex along without intrasm_engine will trigger libc10.so not found error
import torch
from intrasm_engine.pytorch.cpp_extensions.cuda_graph_constructor import (
    TorchCUDAGraphConstructor,
)

from intrasm_engine.common import cutlass_utils
from cuda import cuda


def test_replay_torch():
    # intrasm_engine.pytorch.cpp_extensions.cuda_graph_constructor
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    c_ref = torch.matmul(a, b)
    constructor = TorchCUDAGraphConstructor()
    constructor.capture_library_call_begin()
    c = torch.matmul(a, b)
    constructor.capture_library_call_end()
    constructor.execute_graph()
    constructor.synchronize()
    assert torch.allclose(c, c_ref)


def test_replay_cutlass():
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
        plan, As, Bs, Cs, Ds, print_module=True
    )
    arguments.stream = cuda.CUstream(
        init_value=torch.cuda.current_stream().cuda_stream
    )
    constructor = TorchCUDAGraphConstructor()
    constructor.capture_library_call_begin()
    plan.operation.run(arguments)
    constructor.capture_library_call_end()

    constructor.execute_graph()
    constructor.synchronize()
    Ds_torch = [a @ b for a, b in zip(As, Bs)]

    for d, d_torch in zip(Ds, Ds_torch):
        assert torch.allclose(d, d_torch)


if __name__ == "__main__":
    test_replay_torch()
    test_replay_cutlass()
