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


def gemm_tflops(m, n, k, msec):
    # From flops() at https://github.com/NVIDIA/cutlass/blob/b4b5b110704f4d706a78b190ffadf0e4a86f8289/tools/profiler/src/gemm_operation_profiler.cu
    return 2 * (m * n * k + m * n) / (msec * 1e-3) / 1e12


def test_triton_simt_and_matmul_interleave(
    m=1280, n=1024, k=256, mt=1280, nt=1024, kt=1024
):
    def test_non_interleaving(a, b, matmul_func):
        constructor = TorchCUDAGraphConstructor()
        constructor.capture_library_call_begin()
        c = matmul_func(a, b)
        constructor.capture_library_call_end()
        constructor.instantiate_graph()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        constructor.execute_graph()
        end_event.record()
        constructor.synchronize()
        return start_event.elapsed_time(end_event)

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
        constructor.instantiate_graph()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        constructor.execute_graph()
        end_event.record()
        constructor.synchronize()
        print(f"Interleaved Time event: {start_event.elapsed_time(end_event)}")
        print(
            "TFLOPs",
            gemm_tflops(m, n, k, start_event.elapsed_time(end_event))
            + gemm_tflops(mt, nt, kt, start_event.elapsed_time(end_event)),
        )

    tensor_core_time = test_non_interleaving(  # Tensor core
        torch.randn(mt, kt, device="cuda", dtype=torch.float16),
        torch.randn(kt, nt, device="cuda", dtype=torch.float16),
        torch.matmul,  # TODO: This cannot be the first torch.matmul invoked
    )
    simt_time = test_non_interleaving(  # SIMT
        torch.randn(m, k, device="cuda"),
        torch.randn(k, n, device="cuda"),
        triton_utils.run_matmul,
    )
    print(
        "Non-interleaved time event: ",
        tensor_core_time,
        simt_time,
        tensor_core_time + simt_time,
    )
    print(
        "TFLOPs",
        gemm_tflops(mt, nt, kt, tensor_core_time),
        gemm_tflops(m, n, k, simt_time),
        gemm_tflops(m, n, k, tensor_core_time + simt_time)
        + gemm_tflops(mt, nt, kt, tensor_core_time + simt_time),
    )
    test_interleaving()


if __name__ == "__main__":
    test_triton_matmul()
    test_triton_matmul_fp16()
    test_matmul_fp16()
    test_triton_simt_and_matmul_interleave()
