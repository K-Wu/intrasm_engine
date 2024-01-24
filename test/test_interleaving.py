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


def gemm_tflops(m, n, k, msec):
    #  The flops() at https://github.com/NVIDIA/cutlass/blob/b4b5b110704f4d706a78b190ffadf0e4a86f8289/tools/profiler/src/gemm_operation_profiler.cu seems to count A*B+C. We count A*B only.
    return 2 * (m * n * k) / (msec * 1e-3) / 1e12


def get_matmul_execs_torch_tensor_core(
    mt, nt, kt, repeat_times
) -> list[partial]:
    As = [
        torch.randn(mt, kt, device="cuda", dtype=torch.float16)
        for _ in range(repeat_times)
    ]
    Bs = [
        torch.randn(kt, nt, device="cuda", dtype=torch.float16)
        for _ in range(repeat_times)
    ]
    Cs = [
        torch.zeros((mt, nt), device="cuda", dtype=torch.float16)
        for _ in range(repeat_times)
    ]
    matmul_execs_tc = []
    for idx in range(repeat_times):
        matmul_execs_tc.append(
            partial(torch.matmul, As[idx], Bs[idx], out=Cs[idx])
        )
    return matmul_execs_tc


def _get_matmul_execs_cutlass_tensor_core_f32(
    mt, nt, kt, repeat_times, tile_description: dict[str, Any] | None = None
) -> list[partial]:
    As = [
        torch.randn(mt, kt, device="cuda", dtype=torch.float32)
        for _ in range(repeat_times)
    ]
    Bs = [
        torch.randn(kt, nt, device="cuda", dtype=torch.float32)
        for _ in range(repeat_times)
    ]
    Cs = [
        torch.randn(mt, nt, device="cuda", dtype=torch.float32)
        for _ in range(repeat_times)
    ]
    plan_tcs = [
        cutlass.op.Gemm(
            element=torch.float32,
            layout_A=cutlass.LayoutType.ColumnMajor,
            layout_B=cutlass.LayoutType.RowMajor,
            layout_C=cutlass.LayoutType.ColumnMajor,
            element_C=cutlass.DataType.void,
            element_accumulator=cutlass.DataType.f32,
        )
        for _ in range(repeat_times)
    ]
    # for td in plan_tcs[0].tile_descriptions():
    #     print(td)
    if tile_description is not None:
        for idx in range(repeat_times):
            plan_tcs[idx].tile_description = tile_description
    arguments_tcs = [
        cutlass_utils.prepare_GemmArguments(
            plan_tcs[idx],
            As[idx],
            Bs[idx],
            None,
            Cs[idx],
            print_module=False,
            stream=cuda.CUstream(
                init_value=torch.cuda.current_stream().cuda_stream
            ),
        )
        for idx in range(repeat_times)
    ]
    matmul_execs = [
        partial(plan_tcs[idx].operation.run, arguments_tcs[idx])
        for idx in range(repeat_times)
    ]
    return matmul_execs


def get_matmul_execs_cutlass_tensor_core_f32(
    mt, nt, kt, repeat_times
) -> list[partial]:
    return _get_matmul_execs_cutlass_tensor_core_f32(
        mt,
        nt,
        kt,
        repeat_times,
        {
            "threadblock_shape": [256, 128, 32],
            "warp_count": [4, 2, 1],
            "stages": 3,
        },
    )


def get_matmul_execs_cutlass_tensor_core_f32_small(
    mt, nt, kt, repeat_times
) -> list[partial]:
    return _get_matmul_execs_cutlass_tensor_core_f32(
        mt,
        nt,
        kt,
        repeat_times,
        {
            "threadblock_shape": [64, 64, 32],
            "warp_count": [2, 2, 1],
            "stages": 3,
        },
    )


def _get_matmul_execs_cutlass_tensor_core(
    mt, nt, kt, repeat_times, tile_description: dict[str, Any] | None = None
) -> list[partial]:
    As = [
        torch.randn(mt, kt, device="cuda", dtype=torch.float16)
        for _ in range(repeat_times)
    ]
    Bs = [
        torch.randn(kt, nt, device="cuda", dtype=torch.float16)
        for _ in range(repeat_times)
    ]
    Cs = [
        torch.randn(mt, nt, device="cuda", dtype=torch.float16)
        for _ in range(repeat_times)
    ]
    plan_tcs = [
        cutlass.op.Gemm(
            element=torch.float16,
            layout=cutlass.LayoutType.RowMajor,
            element_C=cutlass.DataType.void,
            element_accumulator=cutlass.DataType.f16,
        )
        for _ in range(repeat_times)
    ]
    if tile_description is not None:
        for idx in range(repeat_times):
            plan_tcs[idx].tile_description = tile_description
    arguments_tcs = [
        cutlass_utils.prepare_GemmArguments(
            plan_tcs[idx],
            As[idx],
            Bs[idx],
            None,
            Cs[idx],
            print_module=False,
            stream=cuda.CUstream(
                init_value=torch.cuda.current_stream().cuda_stream
            ),
        )
        for idx in range(repeat_times)
    ]
    matmul_execs = [
        partial(plan_tcs[idx].operation.run, arguments_tcs[idx])
        for idx in range(repeat_times)
    ]
    return matmul_execs


def get_matmul_execs_cutlass_tensor_core(
    mt, nt, kt, repeat_times
) -> list[partial]:
    return _get_matmul_execs_cutlass_tensor_core(
        mt,
        nt,
        kt,
        repeat_times,
        {
            "threadblock_shape": [128, 256, 32],
            "warp_count": [2, 4, 1],
            "stages": 3,
        },
    )


def get_matmul_execs_cutlass_tensor_core_small(
    mt, nt, kt, repeat_times
) -> list[partial]:
    return _get_matmul_execs_cutlass_tensor_core(
        mt,
        nt,
        kt,
        repeat_times,
        {
            "threadblock_shape": [128, 128, 64],
            "warp_count": [2, 2, 1],
            "stages": 4,
        },
    )


def _get_matmul_execs_cutlass_simt_f32(
    m, n, k, repeat_times, tile_description: dict[str, Any] | None = None
) -> list[partial]:
    As = [torch.randn(m, k, device="cuda") for _ in range(repeat_times)]
    Bs = [torch.randn(k, n, device="cuda") for _ in range(repeat_times)]
    Cs = [torch.randn(m, n, device="cuda") for _ in range(repeat_times)]
    plan_simts = [
        cutlass.op.Gemm(
            element=torch.float32,
            layout_A=cutlass.LayoutType.ColumnMajor,
            layout_B=cutlass.LayoutType.RowMajor,
            layout_C=cutlass.LayoutType.ColumnMajor,
            element_C=cutlass.DataType.void,
            element_accumulator=cutlass.DataType.f32,
        )
        for _ in range(repeat_times)
    ]
    for idx in range(repeat_times):
        plan_simts[idx].opclass = cutlass.OpcodeClass.Simt
        if tile_description is not None:
            plan_simts[idx].tile_description = tile_description
    arguments_tcs = [
        cutlass_utils.prepare_GemmArguments(
            plan_simts[idx],
            As[idx],
            Bs[idx],
            None,
            Cs[idx],
            print_module=False,
            stream=cuda.CUstream(
                init_value=torch.cuda.current_stream().cuda_stream
            ),
        )
        for idx in range(repeat_times)
    ]
    matmul_execs = [
        partial(plan_simts[idx].operation.run, arguments_tcs[idx])
        for idx in range(repeat_times)
    ]
    return matmul_execs


def get_matmul_execs_cutlass_simt_f32(m, n, k, repeat_times) -> list[partial]:
    return _get_matmul_execs_cutlass_simt_f32(
        m,
        n,
        k,
        repeat_times,
        {
            "threadblock_shape": [64, 64, 16],
            "warp_count": [2, 2, 1],
            "stages": 3,
        },
    )


def get_matmul_execs_triton_simt(
    m, n, k, num_nodes_repetition
) -> list[partial]:
    As = [
        torch.randn(m, k, device="cuda", dtype=torch.float16)
        for _ in range(num_nodes_repetition)
    ]
    Bs = [
        torch.randn(k, n, device="cuda", dtype=torch.float16)
        for _ in range(num_nodes_repetition)
    ]
    Cs = [
        torch.zeros((m, n), device="cuda", dtype=torch.float16)
        for _ in range(num_nodes_repetition)
    ]

    matmul_execs_simt = []
    for idx in range(num_nodes_repetition):
        matmul_execs_simt.append(
            partial(
                triton_utils.run_matmul,
                As[idx],
                Bs[idx],
                Cs[idx],
                tiling=MatmulTiling(
                    128,
                    64,
                    64,
                    1,
                    MatrixLayout.COLUMN_MAJOR,
                    MatrixLayout.ROW_MAJOR,
                    MatrixLayout.ROW_MAJOR,
                    2,
                    4,
                ),
            )
        )
    return matmul_execs_simt


def get_matmul_execs_triton_simt_f32(
    m, n, k, num_nodes_repetition
) -> list[partial]:
    As = [
        torch.randn(m, k, device="cuda") for _ in range(num_nodes_repetition)
    ]
    Bs = [
        torch.randn(k, n, device="cuda") for _ in range(num_nodes_repetition)
    ]
    Cs = [
        torch.zeros((m, n), device="cuda") for _ in range(num_nodes_repetition)
    ]

    matmul_execs_simt = []
    for idx in range(num_nodes_repetition):
        matmul_execs_simt.append(
            partial(
                triton_utils.run_matmul,
                As[idx],
                Bs[idx],
                Cs[idx],
                tiling=MatmulTiling(
                    128,
                    64,
                    64,
                    1,
                    MatrixLayout.COLUMN_MAJOR,
                    MatrixLayout.ROW_MAJOR,
                    MatrixLayout.ROW_MAJOR,
                    2,
                    4,
                ),
            )
        )
    return matmul_execs_simt


def run_non_interleaving(
    *matmul_execs_args: list[partial],  # list of partial functions
) -> float:
    repeat_times = len(matmul_execs_args[0])
    Cs = []
    constructor = TorchCUDAGraphConstructor()
    constructor.capture_library_call_begin()
    for idx in range(repeat_times):
        for idx_args in range(len(matmul_execs_args)):
            c = matmul_execs_args[idx_args][idx]()
        Cs.append(c)
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


def run_single_stream(
    *matmul_execs_args: list[partial],  # list of partial functions
) -> float:
    repeat_times = len(matmul_execs_args[0])
    Cs = []
    constructor = TorchCUDAGraphConstructor()
    constructor.capture_library_call_begin()
    for idx in range(repeat_times):
        for idx_args in range(len(matmul_execs_args)):
            c = matmul_execs_args[idx_args][idx]()
        Cs.append(c)
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


def run_torch_interleaving(
    *matmul_execs_args: list[partial], streams: list[CompoundStream]
) -> float:
    assert len(streams) == len(matmul_execs_args)
    repeat_times = len(matmul_execs_args[0])
    results = [[] for _ in range(len(matmul_execs_args))]
    g = torch.cuda.CUDAGraph()
    assert streams[0].torch_stream.stream == torch.cuda.current_stream()
    join_events = [torch.cuda.Event() for _ in range(len(streams))]
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for idx_stream in range(len(streams)):
            with streams[idx_stream].torch_stream as cm:
                for idx in range(repeat_times):
                    # matmul_exec returns the output tensor. Append it in a list to avoid printing.
                    results[idx_stream].append(
                        matmul_execs_args[idx_stream][idx]()
                    )
                if idx_stream > 0:
                    join_events[idx_stream].record(
                        stream=streams[idx_stream].torch_stream.stream
                    )
                    join_events[idx_stream].wait(
                        stream=streams[0].torch_stream.stream
                    )
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()

    start_event.record()
    g.replay()
    end_event.record()
    torch.cuda.synchronize()
    start_event.elapsed_time(end_event)
    print(start_event.elapsed_time(end_event))
    return start_event.elapsed_time(end_event)


def run_interleaving(
    *matmul_execs_args: list[partial], constructor: TorchCUDAGraphConstructor
) -> float:
    # assert len(matmul_execs_args) == 2
    assert len(constructor.registeredStreams) == len(matmul_execs_args)
    repeat_times = len(matmul_execs_args[0])
    results = [[] for _ in range(len(matmul_execs_args))]
    HORIZONTAL_GRAINED = False
    if HORIZONTAL_GRAINED:
        for idx_stream in range(len(constructor.registeredStreams)):
            with constructor.registeredStreams[idx_stream].torch_stream as cm:
                for idx in range(repeat_times):
                    constructor.capture_library_call_begin()
                    # matmul_exec returns the output tensor. Append it in a list to avoid printing.
                    results[idx_stream].append(
                        matmul_execs_args[idx_stream][idx]()
                    )
                    constructor.capture_library_call_end()
    else:
        for idx_stream in range(len(constructor.registeredStreams)):
            with constructor.registeredStreams[idx_stream].torch_stream as cm:
                constructor.capture_library_call_begin()
                for idx in range(repeat_times):
                    # matmul_exec returns the output tensor. Append it in a list to avoid printing.
                    results[idx_stream].append(
                        matmul_execs_args[idx_stream][idx]()
                    )
                constructor.capture_library_call_end()
    constructor.instantiate_graph_exec()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()

    start_event.record()
    constructor.execute_graph()
    end_event.record()
    constructor.synchronize()
    start_event.elapsed_time(end_event)
    constructor.destroy_graph_exec()
    return start_event.elapsed_time(end_event)


def test_triton_simt_and_torch_tensorop_interleave(
    mt=1280, nt=1024, kt=1024, m=1280, n=1024, k=256, num_nodes_repetition=8
):
    matmul_execs_tc = get_matmul_execs_torch_tensor_core(
        mt, nt, kt, num_nodes_repetition
    )
    matmul_execs_simt = get_matmul_execs_triton_simt(
        m, n, k, num_nodes_repetition
    )

    tensor_core_time = run_non_interleaving(  # Tensor core
        matmul_execs_tc,  # FIXME: This cannot be the first torch.matmul invoked; otherwise cublas is not initialized. The current workaround is to print(torch.cuda.current_blas_handle()) before calling this function.
    )

    simt_time = run_non_interleaving(  # SIMT
        matmul_execs_simt,
    )
    print(
        "Separate time event: ",
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

    matmul_execs_tc = get_matmul_execs_torch_tensor_core(
        mt, nt, kt, num_nodes_repetition
    )
    matmul_execs_simt = get_matmul_execs_triton_simt(
        m, n, k, num_nodes_repetition
    )

    constructor = TorchCUDAGraphConstructor()
    constructor.register_new_stream()
    interleave_time = run_interleaving(
        matmul_execs_tc, matmul_execs_simt, constructor=constructor
    )
    print(f"Interleaved Time event: {interleave_time}")
    print(
        "TFLOPs",
        num_nodes_repetition
        * (
            gemm_tflops(m, n, k, interleave_time)
            + gemm_tflops(mt, nt, kt, interleave_time)
        ),
    )


def test_cutlass_simt_f32_and_tensorop_interleave(
    # mt=2560, nt=1024, kt=512, m=1280, n=512, k=512
    mt=1280,
    nt=1024,
    kt=256,
    m=1280,
    n=512,
    k=256,
    repeat_times=8,
):
    matmul_execs_tc = get_matmul_execs_cutlass_tensor_core(
        mt, nt, kt, repeat_times
    )
    matmul_execs_simt = get_matmul_execs_cutlass_simt_f32(
        m, n, k, repeat_times
    )
    tensor_core_time = run_non_interleaving(matmul_execs_tc)
    simt_time = run_non_interleaving(matmul_execs_simt)

    # tensor_core_time = test_canonical_procedure(  # Tensor core
    #     do_simt=False, do_tensor_core=True
    # )
    # simt_time = test_canonical_procedure(  # SIMT
    #     do_simt=True, do_tensor_core=False
    # )
    print(
        "Separate time event: ",
        tensor_core_time,
        simt_time,
        tensor_core_time + simt_time,
    )
    print(
        "TFLOPs",
        repeat_times * gemm_tflops(mt, nt, kt, tensor_core_time),
        repeat_times * gemm_tflops(m, n, k, simt_time),
        repeat_times
        * (
            gemm_tflops(m, n, k, tensor_core_time + simt_time)
            + gemm_tflops(mt, nt, kt, tensor_core_time + simt_time)
        ),
    )

    constructor = TorchCUDAGraphConstructor()
    constructor.register_new_stream()
    matmul_execs_tc = get_matmul_execs_cutlass_tensor_core(
        mt, nt, kt, repeat_times
    )
    with constructor.registeredStreams[1].torch_stream as cm:
        matmul_execs_simt = get_matmul_execs_cutlass_simt_f32(
            m, n, k, repeat_times
        )

    interleaving_time = run_interleaving(
        matmul_execs_tc, matmul_execs_simt, constructor=constructor
    )

    # interleaving_time = test_canonical_procedure(
    #     do_simt=True, do_tensor_core=True
    # )
    print(
        "Interleaved time event: ",
        interleaving_time,
    )
    print(
        "TFLOPs",
        repeat_times
        * (
            gemm_tflops(m, n, k, interleaving_time)
            + gemm_tflops(mt, nt, kt, interleaving_time)
        ),
    )


def test_triton_simt_and_cutlass_tensorop_interleave(
    # mt=2560, nt=1024, kt=512, m=1280, n=512, k=512
    mt=128 * 27,
    nt=1024,
    kt=4096,
    m=(3456 + 6912 - 2 * 128 * 27),
    n=256,
    k=256,
    repeat_times=8,
):
    matmul_execs_tc = get_matmul_execs_cutlass_tensor_core(
        mt, nt, kt, repeat_times
    )
    matmul_execs_simt = get_matmul_execs_triton_simt(m, n, k, repeat_times)
    tensor_core_time = run_non_interleaving(matmul_execs_tc)
    simt_time = run_non_interleaving(matmul_execs_simt)
    # tensor_core_time = test_canonical_procedure(  # Tensor core
    #     do_simt=False, do_tensor_core=True
    # )
    # simt_time = test_canonical_procedure(  # SIMT
    #     do_simt=True, do_tensor_core=False
    # )
    print(
        "Separate time event: ",
        tensor_core_time,
        simt_time,
        tensor_core_time + simt_time,
    )
    print(
        "TFLOPs",
        repeat_times * gemm_tflops(mt, nt, kt, tensor_core_time),
        repeat_times * gemm_tflops(m, n, k, simt_time),
        repeat_times
        * (
            gemm_tflops(m, n, k, tensor_core_time + simt_time)
            + gemm_tflops(mt, nt, kt, tensor_core_time + simt_time)
        ),
    )

    constructor = TorchCUDAGraphConstructor()
    constructor.register_new_stream()
    matmul_execs_tc = get_matmul_execs_cutlass_tensor_core(
        mt, nt, kt, repeat_times
    )
    with constructor.registeredStreams[1].torch_stream as cm:
        matmul_execs_simt = get_matmul_execs_triton_simt(m, n, k, repeat_times)

    interleaving_time = run_interleaving(
        matmul_execs_tc, matmul_execs_simt, constructor=constructor
    )
    # interleaving_time = test_canonical_procedure(
    #     do_simt=True, do_tensor_core=True
    # )
    print(
        "Interleaved time event: ",
        interleaving_time,
    )
    print(
        "TFLOPs",
        repeat_times
        * (
            gemm_tflops(m, n, k, interleaving_time)
            + gemm_tflops(mt, nt, kt, interleaving_time)
        ),
    )


def test_triton_simt_f32_and_cutlass_tensorop_f32_interleave(
    # A100
    # mt=128 * 27,
    # nt=1024,
    # kt=4096,
    # m=(3456 + 6912 - 2 * 128 * 27),
    # n=256,
    # k=256,
    # RTX 3090
    mt=128 * 41,
    nt=256,
    kt=4096,
    m=128 * 41,
    n=64 * 2,
    k=1024,
    repeat_times=8,
):
    matmul_execs_tc = get_matmul_execs_cutlass_tensor_core_f32(
        mt, nt, kt, repeat_times
    )
    matmul_execs_simt = get_matmul_execs_triton_simt_f32(m, n, k, repeat_times)
    tensor_core_time = run_non_interleaving(matmul_execs_tc)
    simt_time = run_non_interleaving(matmul_execs_simt)
    # tensor_core_time = test_canonical_procedure(  # Tensor core
    #     do_simt=False, do_tensor_core=True
    # )
    # simt_time = test_canonical_procedure(  # SIMT
    #     do_simt=True, do_tensor_core=False
    # )
    print(
        "Separate time event: ",
        tensor_core_time,
        simt_time,
        tensor_core_time + simt_time,
    )
    print(
        "TFLOPs",
        repeat_times * gemm_tflops(mt, nt, kt, tensor_core_time),
        repeat_times * gemm_tflops(m, n, k, simt_time),
        repeat_times
        * (
            gemm_tflops(m, n, k, tensor_core_time + simt_time)
            + gemm_tflops(mt, nt, kt, tensor_core_time + simt_time)
        ),
    )

    constructor = TorchCUDAGraphConstructor()
    constructor.register_new_stream()
    matmul_execs_tc = get_matmul_execs_cutlass_tensor_core_f32(
        mt, nt, kt, repeat_times
    )
    with constructor.registeredStreams[1].torch_stream as cm:
        matmul_execs_simt = get_matmul_execs_triton_simt_f32(
            m, n, k, repeat_times
        )

    interleaving_time = run_interleaving(
        matmul_execs_tc, matmul_execs_simt, constructor=constructor
    )
    # interleaving_time = test_canonical_procedure(
    #     do_simt=True, do_tensor_core=True
    # )
    print(
        "Interleaved time event: ",
        interleaving_time,
    )
    print(
        "TFLOPs",
        repeat_times
        * (
            gemm_tflops(m, n, k, interleaving_time)
            + gemm_tflops(mt, nt, kt, interleaving_time)
        ),
    )


def test_cutlass_simt_f32_and_tensorop_f32_interleave(
    # A100
    mt=256 * 27,
    nt=128 * 4,
    kt=4096,
    m=64 * 27 * 2,
    n=64 * 2,
    k=512,
    # RTX 3090
    # mt=64 * 41,
    # nt=64 * 2,
    # kt=4096,
    # m=64 * 41,
    # n=64 * 2,
    # k=1024,
    repeat_times=8,
):
    matmul_execs_tc = get_matmul_execs_cutlass_tensor_core_f32(
        mt, nt, kt, repeat_times
    )
    matmul_execs_simt = get_matmul_execs_cutlass_simt_f32(
        m, n, k, repeat_times
    )
    tensor_core_time = run_non_interleaving(matmul_execs_tc)
    simt_time = run_non_interleaving(matmul_execs_simt)
    # tensor_core_time = test_canonical_procedure(  # Tensor core
    #     do_simt=False, do_tensor_core=True
    # )
    # simt_time = test_canonical_procedure(  # SIMT
    #     do_simt=True, do_tensor_core=False
    # )
    print(
        "Separate time event: ",
        tensor_core_time,
        simt_time,
        tensor_core_time + simt_time,
    )
    print(
        "TFLOPs",
        repeat_times * gemm_tflops(mt, nt, kt, tensor_core_time),
        repeat_times * gemm_tflops(m, n, k, simt_time),
        repeat_times
        * (
            gemm_tflops(m, n, k, tensor_core_time + simt_time)
            + gemm_tflops(mt, nt, kt, tensor_core_time + simt_time)
        ),
    )

    non_interleave_time = run_non_interleaving(
        matmul_execs_tc, matmul_execs_simt
    )
    print(
        "Non-interleaved serial time event: ",
        non_interleave_time,
    )
    print(
        "TFLOPs",
        repeat_times
        * (
            gemm_tflops(m, n, k, non_interleave_time)
            + gemm_tflops(mt, nt, kt, non_interleave_time)
        ),
    )

    single_stream_time = run_single_stream(matmul_execs_tc, matmul_execs_simt)
    print(
        "Single stream serial time event: ",
        single_stream_time,
    )
    print(
        "TFLOPs",
        repeat_times
        * (
            gemm_tflops(m, n, k, single_stream_time)
            + gemm_tflops(mt, nt, kt, single_stream_time)
        ),
    )

    constructor = TorchCUDAGraphConstructor()
    constructor.register_new_stream()
    matmul_execs_tc = get_matmul_execs_cutlass_tensor_core_f32(
        mt, nt, kt, repeat_times
    )
    with constructor.registeredStreams[1].torch_stream as cm:
        matmul_execs_simt = get_matmul_execs_cutlass_simt_f32(
            m, n, k, repeat_times
        )

    interleaving_time = run_interleaving(
        matmul_execs_tc, matmul_execs_simt, constructor=constructor
    )
    # interleaving_time = test_canonical_procedure(
    #     do_simt=True, do_tensor_core=True
    # )
    print(
        "Interleaved time event: ",
        interleaving_time,
    )
    print(
        "TFLOPs",
        repeat_times
        * (
            gemm_tflops(m, n, k, interleaving_time)
            + gemm_tflops(mt, nt, kt, interleaving_time)
        ),
    )

    matmul_execs_tc = get_matmul_execs_cutlass_tensor_core_f32(
        mt, nt, kt, repeat_times
    )
    with constructor.registeredStreams[1].torch_stream as cm:
        matmul_execs_simt = get_matmul_execs_cutlass_simt_f32(
            m, n, k, repeat_times
        )

    TORCH_GRAPH_FAULTY = True
    if not TORCH_GRAPH_FAULTY:
        torch_graph_time = run_torch_interleaving(
            matmul_execs_tc,
            matmul_execs_simt,
            streams=constructor.registeredStreams,
        )
        print(
            "Torch graph time event: ",
            torch_graph_time,
        )
        print(
            "TFLOPs",
            repeat_times
            * (
                gemm_tflops(m, n, k, torch_graph_time)
                + gemm_tflops(mt, nt, kt, torch_graph_time)
            ),
        )


def test_cutlass_tensorop_f32_big_and_small_interleave(
    # A100
    mt=256 * 27,
    nt=128 * 4,
    kt=4096,
    m=64 * 27 * 2,
    n=64 * 2,
    k=1024,
    # RTX 3090
    # mt=64 * 41,
    # nt=64 * 2,
    # kt=4096,
    # m=64 * 41,
    # n=64 * 2,
    # k=1024,
    repeat_times=8,
):
    matmul_execs_tc = get_matmul_execs_cutlass_tensor_core_f32(
        mt, nt, kt, repeat_times
    )
    matmul_execs_simt = get_matmul_execs_cutlass_tensor_core_f32_small(
        m, n, k, repeat_times
    )
    tensor_core_time = run_non_interleaving(matmul_execs_tc)
    simt_time = run_non_interleaving(matmul_execs_simt)
    # tensor_core_time = test_canonical_procedure(  # Tensor core
    #     do_simt=False, do_tensor_core=True
    # )
    # simt_time = test_canonical_procedure(  # SIMT
    #     do_simt=True, do_tensor_core=False
    # )
    print(
        "Separate time event: ",
        tensor_core_time,
        simt_time,
        tensor_core_time + simt_time,
    )
    print(
        "TFLOPs",
        repeat_times * gemm_tflops(mt, nt, kt, tensor_core_time),
        repeat_times * gemm_tflops(m, n, k, simt_time),
        repeat_times
        * (
            gemm_tflops(m, n, k, tensor_core_time + simt_time)
            + gemm_tflops(mt, nt, kt, tensor_core_time + simt_time)
        ),
    )

    non_interleave_time = run_non_interleaving(
        matmul_execs_tc, matmul_execs_simt
    )
    print(
        "Non-interleaved serial time event: ",
        non_interleave_time,
    )
    print(
        "TFLOPs",
        repeat_times
        * (
            gemm_tflops(m, n, k, non_interleave_time)
            + gemm_tflops(mt, nt, kt, non_interleave_time)
        ),
    )

    single_stream_time = run_single_stream(matmul_execs_tc, matmul_execs_simt)
    print(
        "Single stream serial time event: ",
        single_stream_time,
    )
    print(
        "TFLOPs",
        repeat_times
        * (
            gemm_tflops(m, n, k, single_stream_time)
            + gemm_tflops(mt, nt, kt, single_stream_time)
        ),
    )

    constructor = TorchCUDAGraphConstructor()
    constructor.register_new_stream()
    matmul_execs_tc = get_matmul_execs_cutlass_tensor_core_f32(
        mt, nt, kt, repeat_times
    )
    with constructor.registeredStreams[1].torch_stream as cm:
        matmul_execs_simt = get_matmul_execs_cutlass_tensor_core_f32_small(
            m, n, k, repeat_times
        )

    interleaving_time = run_interleaving(
        matmul_execs_tc, matmul_execs_simt, constructor=constructor
    )
    # interleaving_time = test_canonical_procedure(
    #     do_simt=True, do_tensor_core=True
    # )
    print(
        "Interleaved time event: ",
        interleaving_time,
    )
    print(
        "TFLOPs",
        repeat_times
        * (
            gemm_tflops(m, n, k, interleaving_time)
            + gemm_tflops(mt, nt, kt, interleaving_time)
        ),
    )

    matmul_execs_tc = get_matmul_execs_cutlass_tensor_core_f32(
        mt, nt, kt, repeat_times
    )
    with constructor.registeredStreams[1].torch_stream as cm:
        matmul_execs_simt = get_matmul_execs_cutlass_tensor_core_f32_small(
            m, n, k, repeat_times
        )

    TORCH_GRAPH_FAULTY = True
    if not TORCH_GRAPH_FAULTY:
        torch_graph_time = run_torch_interleaving(
            matmul_execs_tc,
            matmul_execs_simt,
            streams=constructor.registeredStreams,
        )
        print(
            "Torch graph time event: ",
            torch_graph_time,
        )
        print(
            "TFLOPs",
            repeat_times
            * (
                gemm_tflops(m, n, k, torch_graph_time)
                + gemm_tflops(mt, nt, kt, torch_graph_time)
            ),
        )


def test_cutlass_tensorop_small_and_small_interleave(
    # mt=2560, nt=1024, kt=512, m=1280, n=512, k=512
    mb=128 * 27 * 8,
    # nb=2048,
    nb=1536,
    kb=4096,
    m=128 * 27,  # 3456 + 6912 -2 * 128*27,
    n=512,
    k=1024,
    repeat_times=8,
):
    matmul_execs_tc_big = get_matmul_execs_cutlass_tensor_core_small(
        mb, nb, kb, repeat_times
    )
    matmul_execs_tc = get_matmul_execs_cutlass_tensor_core_small(
        m, n, k, repeat_times
    )
    tensor_core_big_time = run_non_interleaving(matmul_execs_tc_big)
    tensor_core_time = run_non_interleaving(matmul_execs_tc)
    # tensor_core_time = test_canonical_procedure(  # Tensor core
    #     do_small=False, do_big=True
    # )
    # simt_time = test_canonical_procedure(do_small=True, do_big=False)  # SIMT
    print(
        "Separate time event: ",
        tensor_core_big_time,
        tensor_core_time,
        tensor_core_big_time + tensor_core_time,
    )
    print(
        "TFLOPs",
        repeat_times * gemm_tflops(mb, nb, kb, tensor_core_big_time),
        repeat_times * gemm_tflops(m, n, k, tensor_core_time),
        repeat_times
        * (
            gemm_tflops(m, n, k, tensor_core_big_time + tensor_core_time)
            + gemm_tflops(mb, nb, kb, tensor_core_big_time + tensor_core_time)
        ),
    )

    constructor = TorchCUDAGraphConstructor()
    constructor.register_new_stream()
    matmul_execs_tc_big = get_matmul_execs_cutlass_tensor_core_small(
        mb, nb, kb, repeat_times
    )
    with constructor.registeredStreams[1].torch_stream as cm:
        matmul_execs_tc = get_matmul_execs_cutlass_tensor_core_small(
            m, n, k, repeat_times
        )

    interleaving_time = run_interleaving(
        matmul_execs_tc_big, matmul_execs_tc, constructor=constructor
    )
    # interleaving_time = test_canonical_procedure(do_small=True, do_big=True)
    print(
        "Interleaved time event: ",
        interleaving_time,
    )
    print(
        "TFLOPs",
        repeat_times
        * (
            gemm_tflops(m, n, k, interleaving_time)
            + gemm_tflops(mb, nb, kb, interleaving_time)
        ),
    )


def test_cutlass_tensorop_big_and_small_interleave(
    # mt=2560, nt=1024, kt=512, m=1280, n=512, k=512
    mb=128 * 27 * 8,
    # nb=2048,
    nb=1024,
    kb=4096,
    m=128 * 27 * 4,  # 3456 + 6912 -2 * 128*27,
    n=512 * 2,
    k=2048,
    repeat_times=1,
):
    matmul_execs_tc_big = get_matmul_execs_cutlass_tensor_core(
        mb, nb, kb, repeat_times
    )
    matmul_execs_tc = get_matmul_execs_cutlass_tensor_core_small(
        m, n, k, repeat_times
    )
    tensor_core_big_time = run_non_interleaving(matmul_execs_tc_big)
    tensor_core_time = run_non_interleaving(matmul_execs_tc)
    # tensor_core_time = test_canonical_procedure(  # Tensor core
    #     do_small=False, do_big=True
    # )
    # simt_time = test_canonical_procedure(do_small=True, do_big=False)  # SIMT
    print(
        "Separate time event: ",
        tensor_core_big_time,
        tensor_core_time,
        tensor_core_big_time + tensor_core_time,
    )
    print(
        "TFLOPs",
        repeat_times * gemm_tflops(mb, nb, kb, tensor_core_big_time),
        repeat_times * gemm_tflops(m, n, k, tensor_core_time),
        repeat_times
        * (
            gemm_tflops(m, n, k, tensor_core_big_time + tensor_core_time)
            + gemm_tflops(mb, nb, kb, tensor_core_big_time + tensor_core_time)
        ),
    )

    constructor = TorchCUDAGraphConstructor()
    constructor.register_new_stream()
    matmul_execs_tc_big = get_matmul_execs_cutlass_tensor_core(
        mb, nb, kb, repeat_times
    )
    with constructor.registeredStreams[1].torch_stream as cm:
        matmul_execs_tc = get_matmul_execs_cutlass_tensor_core_small(
            m, n, k, repeat_times
        )

    interleaving_time = run_interleaving(
        matmul_execs_tc_big, matmul_execs_tc, constructor=constructor
    )
    # interleaving_time = test_canonical_procedure(do_small=True, do_big=True)
    print(
        "Interleaved time event: ",
        interleaving_time,
    )
    print(
        "TFLOPs",
        repeat_times
        * (
            gemm_tflops(m, n, k, interleaving_time)
            + gemm_tflops(mb, nb, kb, interleaving_time)
        ),
    )


if __name__ == "__main__":
    # print(
    #     "Triton SIMT in parallel to Torch TensorOp. Repeat with different data"
    # )
    # test_triton_simt_and_torch_tensorop_interleave()
    # print("Cutlass SIMT (f32) in parallel to TensorOp. Repeat with different data")
    # test_cutlass_simt_f32_and_tensorop_interleave()
    # print(
    #     "Triton SIMT in parallel to Cutlass TensorOp. Repeat with different"
    #     " data"
    # )
    # test_triton_simt_and_cutlass_tensorop_interleave()
    # print(
    #     "Triton SIMT (f32) in parallel to Cutlass TensorOp (f32). Repeat with"
    #     " different data"
    # )
    # test_triton_simt_f32_and_cutlass_tensorop_f32_interleave()
    print(
        "Cutlass SIMT (f32) in parallel to TensorOp (f32). Repeat with"
        " different data"
    )
    test_cutlass_simt_f32_and_tensorop_f32_interleave()
    print(
        "Cutlass TensorOp big (f32) in parallel to small (f32). Repeat with"
        " different data"
    )
    test_cutlass_tensorop_f32_big_and_small_interleave()
    # print(
    #     "Cutlass TensorOp small in parallel to small. Repeat with different"
    #     " data"
    # )
    # test_cutlass_tensorop_small_and_small_interleave()
    # print(
    #     "Cutlass TensorOp big in parallel to small. Repeat with different data"
    # )
    # test_cutlass_tensorop_big_and_small_interleave()
