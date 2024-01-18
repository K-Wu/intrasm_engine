import torch
import intrasm_engine.common.triton_utils as triton_utils
import intrasm_engine_extensions as iex
from intrasm_engine.pytorch.cpp_extensions.cuda_graph_constructor import (
    TorchCUDAGraphConstructor,
)
import time


def test_triton_matmul():
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    c = triton_utils.run_matmul(a, b)


def test_triton_matmul_fp16():
    a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    b = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    c = triton_utils.run_matmul(a, b)


def test_matmul_fp16():
    a = torch.randn(2048, 64, device="cuda", dtype=torch.float16)
    b = torch.randn(64, 128, device="cuda", dtype=torch.float16)
    c = torch.matmul(a, b)


def test_triton_simt_and_matmul_interleave(
    m=256, n=256, k=128, mt=2048, nt=128, kt=64
):
    def test_non_interleaving(a, b, matmul_func):
        constructor = TorchCUDAGraphConstructor()
        constructor.capture_library_call_begin()
        c = matmul_func(a, b)
        constructor.capture_library_call_end()
        torch.cuda.synchronize()
        start = time.perf_counter()
        constructor.execute_graph()
        constructor.synchronize()
        end = time.perf_counter()
        return end - start

    def test_interleaving():
        a = torch.randn(mt, kt, device="cuda", dtype=torch.float16)
        b = torch.randn(kt, nt, device="cuda", dtype=torch.float16)
        a2 = torch.randn(m, k, device="cuda")
        b2 = torch.randn(k, n, device="cuda")
        constructor = TorchCUDAGraphConstructor()
        constructor.capture_library_call_begin()
        c = torch.matmul(a, b)  # Tensor core
        constructor.capture_library_call_end()
        constructor.capture_library_call_begin()
        c2 = triton_utils.run_matmul(a2, b2)  # SIMT
        constructor.capture_library_call_end()
        torch.cuda.synchronize()
        start = time.perf_counter()
        constructor.execute_graph()
        constructor.synchronize()
        end = time.perf_counter()
        print(f"Interleaved Time: {end - start}")

    print(
        "Non-interleaved time: ",
        test_non_interleaving(  # Tensor core
            torch.randn(mt, kt, device="cuda", dtype=torch.float16),
            torch.randn(kt, nt, device="cuda", dtype=torch.float16),
            torch.matmul,
        )
        + test_non_interleaving(  # SIMT
            torch.randn(m, k, device="cuda"),
            torch.randn(k, n, device="cuda"),
            triton_utils.run_matmul,
        ),
    )
    test_interleaving()


if __name__ == "__main__":
    test_triton_matmul()
    test_triton_matmul_fp16()
    test_matmul_fp16()
    test_triton_simt_and_matmul_interleave()
