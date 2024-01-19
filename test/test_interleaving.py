import torch
import intrasm_engine.common.triton_utils as triton_utils
import intrasm_engine.common.cutlass_utils as cutlass_utils
import intrasm_engine_extensions as iex
from intrasm_engine.pytorch.cpp_extensions.cuda_graph_constructor import (
    TorchCUDAGraphConstructor,
)
import cutlass
from cuda import cuda
import contextlib


def gemm_tflops(m, n, k, msec):
    #  The flops() at https://github.com/NVIDIA/cutlass/blob/b4b5b110704f4d706a78b190ffadf0e4a86f8289/tools/profiler/src/gemm_operation_profiler.cu seems to count A*B+C. We count A*B only.
    return 2 * (m * n * k) / (msec * 1e-3) / 1e12


def test_triton_simt_and_torch_tensorop_interleave(
    mt=1280, nt=1024, kt=1024, m=1280, n=1024, k=256
):
    def test_non_interleaving(a, b, matmul_func):
        constructor = TorchCUDAGraphConstructor()
        constructor.capture_library_call_begin()
        c = matmul_func(a, b)
        constructor.capture_library_call_end()
        constructor.instantiate_graph_exec()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        constructor.execute_graph()
        end_event.record()
        constructor.synchronize()
        constructor.destroy_graph_exec()
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
        constructor.instantiate_graph_exec()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        constructor.execute_graph()
        end_event.record()
        constructor.synchronize()
        constructor.destroy_graph_exec()
        print(f"Interleaved Time event: {start_event.elapsed_time(end_event)}")
        print(
            "TFLOPs",
            gemm_tflops(m, n, k, start_event.elapsed_time(end_event))
            + gemm_tflops(mt, nt, kt, start_event.elapsed_time(end_event)),
        )

    tensor_core_time = test_non_interleaving(  # Tensor core
        torch.randn(mt, kt, device="cuda", dtype=torch.float16),
        torch.randn(kt, nt, device="cuda", dtype=torch.float16),
        torch.matmul,  # FIXME: This cannot be the first torch.matmul invoked; otherwise cublas is not initialized. The current workaround is to print(torch.cuda.current_blas_handle()) before calling this function.
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


def test_cutlass_simt_and_tensorop_interleave(
    # mt=2560, nt=1024, kt=512, m=1280, n=512, k=512
    mt=1280,
    nt=1024,
    kt=256,
    m=1280,
    n=512,
    k=256,
    num_nodes_repetition=32,
):
    def test_canonical_procedure(do_tensor_core: bool, do_simt: bool):
        constructor = TorchCUDAGraphConstructor()
        if do_tensor_core and do_simt:
            constructor.register_new_stream()
        if do_tensor_core:
            a = torch.randn(mt, kt, device="cuda", dtype=torch.float16)
            b = torch.randn(kt, nt, device="cuda", dtype=torch.float16)
            c = torch.randn(mt, nt, device="cuda", dtype=torch.float16)
            plan_tc = cutlass.op.Gemm(
                element=torch.float16,
                layout=cutlass.LayoutType.RowMajor,
                element_C=cutlass.DataType.void,
                element_accumulator=cutlass.DataType.f16,
            )
            # for td in plan_tc.tile_descriptions():
            #     print(td)
            # plan_tc.tile_description = {
            #     "threadblock_shape": [128, 256, 32],
            #     "warp_count": [2, 4, 1],
            #     "stages": 3,
            # }
            plan_tc.tile_description = {
                "threadblock_shape": [64, 128, 32],
                "warp_count": [2, 2, 1],
                "stages": 3,
            }
            arguments_tc = cutlass_utils.prepare_GemmArguments(
                plan_tc,
                a,
                b,
                None,
                c,
                print_module=False,
                stream=cuda.CUstream(
                    init_value=torch.cuda.current_stream().cuda_stream
                ),
            )

        if do_simt:
            a2 = torch.randn(m, k, device="cuda")
            b2 = torch.randn(k, n, device="cuda")
            c2 = torch.randn(m, n, device="cuda")
            plan_simt = cutlass.op.Gemm(
                element=torch.float32,
                layout=cutlass.LayoutType.RowMajor,
                element_C=cutlass.DataType.void,
                element_accumulator=cutlass.DataType.f32,
            )
            plan_simt.opclass = cutlass.OpcodeClass.Simt
            # for td in plan_simt.tile_descriptions():
            #     print(td)
            plan_simt.tile_description = {
                "threadblock_shape": [128, 64, 8],
                "warp_count": [2, 2, 1],
                "stages": 5,
            }
            if do_tensor_core:
                stream = constructor.registeredStreams[
                    1
                ].torch_stream.stream.cuda_stream
            else:
                stream = cuda.CUstream(
                    init_value=torch.cuda.current_stream().cuda_stream
                )
            arguments_simt = cutlass_utils.prepare_GemmArguments(
                plan_simt,
                a2,
                b2,
                None,
                c2,
                print_module=False,
                stream=stream,
            )

        if do_tensor_core:
            for _ in range(num_nodes_repetition):
                constructor.capture_library_call_begin()
                plan_tc.operation.run(arguments_tc)  # Tensor core
                constructor.capture_library_call_end()
        if do_simt:
            if do_tensor_core:
                cm = constructor.registeredStreams[1].torch_stream
            else:  # null cm
                cm = contextlib.nullcontext()
            with cm:
                for _ in range(num_nodes_repetition):
                    constructor.capture_library_call_begin()
                    plan_simt.operation.run(arguments_simt)  # SIMT
                    constructor.capture_library_call_end()

        constructor.instantiate_graph_exec()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        constructor.execute_graph()
        end_event.record()
        constructor.synchronize()
        constructor.destroy_graph_exec()
        return start_event.elapsed_time(end_event)

    tensor_core_time = test_canonical_procedure(  # Tensor core
        do_simt=False, do_tensor_core=True
    )
    simt_time = test_canonical_procedure(  # SIMT
        do_simt=True, do_tensor_core=False
    )
    print(
        "Non-interleaved time event: ",
        tensor_core_time,
        simt_time,
        tensor_core_time + simt_time,
    )
    print(
        "TFLOPs",
        num_nodes_repetition * gemm_tflops(mt, nt, kt, tensor_core_time),
        num_nodes_repetition * gemm_tflops(m, n, k, simt_time),
        num_nodes_repetition
        * (
            gemm_tflops(m, n, k, tensor_core_time + simt_time)
            + gemm_tflops(mt, nt, kt, tensor_core_time + simt_time)
        ),
    )

    interleaving_time = test_canonical_procedure(
        do_simt=True, do_tensor_core=True
    )
    print(
        "Interleaved time event: ",
        interleaving_time,
    )
    print(
        "TFLOPs",
        num_nodes_repetition
        * (
            gemm_tflops(m, n, k, interleaving_time)
            + gemm_tflops(mt, nt, kt, interleaving_time)
        ),
    )


if __name__ == "__main__":
    test_triton_simt_and_torch_tensorop_interleave()
    test_cutlass_simt_and_tensorop_interleave()
