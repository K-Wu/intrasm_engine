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
    mt=1280, nt=1024, kt=1024, m=1280, n=1024, k=256, num_nodes_repetition=8
):
    def test_non_interleaving(
        As: list[torch.Tensor], Bs: list[torch.Tensor], matmul_func
    ):
        Cs = []
        constructor = TorchCUDAGraphConstructor()
        for idx in range(num_nodes_repetition):
            constructor.capture_library_call_begin()
            c = matmul_func(As[idx], Bs[idx])
            constructor.capture_library_call_end()
            Cs.append(c)
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
        As = []
        Bs = []
        Cs = []
        A2s = []
        B2s = []
        C2s = []
        for _ in range(num_nodes_repetition):
            a = torch.randn(mt, kt, device="cuda", dtype=torch.float16)
            b = torch.randn(kt, nt, device="cuda", dtype=torch.float16)
            c = torch.zeros((mt, nt), device="cuda", dtype=torch.float16)
            a2 = torch.randn(m, k, device="cuda")
            b2 = torch.randn(k, n, device="cuda")
            c2 = torch.zeros((m, n), device="cuda")
            As.append(a)
            Bs.append(b)
            Cs.append(c)
            A2s.append(a2)
            B2s.append(b2)
            C2s.append(c2)
        constructor = TorchCUDAGraphConstructor()
        constructor.register_new_stream()
        for idx in range(num_nodes_repetition):
            constructor.capture_library_call_begin()
            # torch.matmul returns the output tensor, i.e., c. Specify it as lhs has no effect but to avoid printing.
            Cs[idx] = torch.matmul(
                As[idx], Bs[idx], out=Cs[idx]
            )  # Tensor core
            constructor.capture_library_call_end()
        with constructor.registeredStreams[1].torch_stream as cm:
            for idx in range(num_nodes_repetition):
                constructor.capture_library_call_begin()
                # run_matmul returns the output tensor, i.e., c2. Specify it as lhs has no effect but explicitly to avoid printing.
                C2s[idx] = triton_utils.run_matmul(
                    A2s[idx], B2s[idx], C2s[idx]
                )  # SIMT
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
            num_nodes_repetition
            * (
                gemm_tflops(m, n, k, start_event.elapsed_time(end_event))
                + gemm_tflops(mt, nt, kt, start_event.elapsed_time(end_event))
            ),
        )

    As = [
        torch.randn(mt, kt, device="cuda", dtype=torch.float16)
        for _ in range(num_nodes_repetition)
    ]
    Bs = [
        torch.randn(kt, nt, device="cuda", dtype=torch.float16)
        for _ in range(num_nodes_repetition)
    ]
    tensor_core_time = test_non_interleaving(  # Tensor core
        As,
        Bs,
        torch.matmul,  # FIXME: This cannot be the first torch.matmul invoked; otherwise cublas is not initialized. The current workaround is to print(torch.cuda.current_blas_handle()) before calling this function.
    )
    As = [
        torch.randn(m, k, device="cuda") for _ in range(num_nodes_repetition)
    ]
    Bs = [
        torch.randn(k, n, device="cuda") for _ in range(num_nodes_repetition)
    ]
    simt_time = test_non_interleaving(  # SIMT
        As,
        Bs,
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
        num_nodes_repetition * gemm_tflops(mt, nt, kt, tensor_core_time),
        num_nodes_repetition * gemm_tflops(m, n, k, simt_time),
        num_nodes_repetition
        * (
            gemm_tflops(m, n, k, tensor_core_time + simt_time)
            + gemm_tflops(mt, nt, kt, tensor_core_time + simt_time)
        ),
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
    num_nodes_repetition=8,
):
    def test_canonical_procedure(do_tensor_core: bool, do_simt: bool):
        As = []
        Bs = []
        Cs = []
        plan_tcs = []
        arguments_tcs = []
        A2s = []
        B2s = []
        C2s = []
        plan_simts = []
        arguments_simts = []
        constructor = TorchCUDAGraphConstructor()
        if do_tensor_core and do_simt:
            constructor.register_new_stream()
        if do_tensor_core:
            for _ in range(num_nodes_repetition):
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
                As.append(a)
                Bs.append(b)
                Cs.append(c)
                plan_tcs.append(plan_tc)
                arguments_tcs.append(arguments_tc)

        if do_simt:
            for _ in range(num_nodes_repetition):
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
                A2s.append(a2)
                B2s.append(b2)
                C2s.append(c2)
                plan_simts.append(plan_simt)
                arguments_simts.append(arguments_simt)

        if do_tensor_core:
            for idx in range(num_nodes_repetition):
                constructor.capture_library_call_begin()
                plan_tcs[idx].operation.run(arguments_tcs[idx])  # Tensor core
                constructor.capture_library_call_end()
        if do_simt:
            if do_tensor_core:
                cm = constructor.registeredStreams[1].torch_stream
            else:  # null cm
                cm = contextlib.nullcontext()
            with cm:
                for idx in range(num_nodes_repetition):
                    constructor.capture_library_call_begin()
                    plan_simts[idx].operation.run(arguments_simts[idx])  # SIMT
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


def test_triton_simt_and_cutlass_tensorop_interleave(
    # mt=2560, nt=1024, kt=512, m=1280, n=512, k=512
    mt=2560,
    nt=1024,
    kt=1572,
    m=1280,
    n=512,
    k=1024,
    num_nodes_repetition=8,
):
    def test_canonical_procedure(do_tensor_core: bool, do_simt: bool):
        As = []
        Bs = []
        Cs = []
        plan_tcs = []
        arguments_tcs = []
        A2s = []
        B2s = []
        C2s = []

        constructor = TorchCUDAGraphConstructor()
        if do_tensor_core and do_simt:
            constructor.register_new_stream()
        if do_tensor_core:
            for _ in range(num_nodes_repetition):
                a = torch.randn(mt, kt, device="cuda", dtype=torch.float16)
                b = torch.randn(kt, nt, device="cuda", dtype=torch.float16)
                c = torch.randn(mt, nt, device="cuda", dtype=torch.float16)
                As.append(a)
                Bs.append(b)
                Cs.append(c)
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

            for _ in range(num_nodes_repetition):
                plan_tc.tile_description = {
                    "threadblock_shape": [128, 256, 32],
                    "warp_count": [2, 4, 1],
                    "stages": 3,
                }
                plan_tcs.append(plan_tc)
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
                arguments_tcs.append(arguments_tc)

        if do_simt:
            for _ in range(num_nodes_repetition):
                a2 = torch.randn(m, k, device="cuda", dtype=torch.float16)
                b2 = torch.randn(k, n, device="cuda", dtype=torch.float16)
                c2 = torch.zeros(m, n, device="cuda", dtype=torch.float16)
                A2s.append(a2)
                B2s.append(b2)
                C2s.append(c2)

        if do_tensor_core:
            for idx_rep in range(num_nodes_repetition):
                constructor.capture_library_call_begin()
                # Tensor core
                plan_tcs[idx_rep].operation.run(arguments_tcs[idx_rep])
                constructor.capture_library_call_end()
        if do_simt:
            if do_tensor_core:
                cm = constructor.registeredStreams[1].torch_stream
            else:  # null cm
                cm = contextlib.nullcontext()
            with cm:
                for idx_rep in range(num_nodes_repetition):
                    constructor.capture_library_call_begin()
                    # SIMT
                    # run_matmul returns the output tensor, i.e., c2. Specify it as lhs has no effect but explicitly to avoid printing.
                    C2s[idx_rep] = triton_utils.run_matmul(
                        A2s[idx_rep], B2s[idx_rep], C2s[idx_rep]
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


def test_cutlass_tensorop_big_and_small_interleave(
    # mt=2560, nt=1024, kt=512, m=1280, n=512, k=512
    mb=2560,
    # nb=2048,
    nb=1024,
    kb=1572,
    m=1280,
    n=512,
    k=1024,
    num_nodes_repetition=8,
):
    def test_canonical_procedure(do_big: bool, do_small: bool):
        As = []
        Bs = []
        Cs = []
        plan_tcs = []
        arguments_tcs = []
        A2s = []
        B2s = []
        C2s = []
        plan_simts = []
        arguments_simts = []
        constructor = TorchCUDAGraphConstructor()
        if do_big and do_small:
            constructor.register_new_stream()
        if do_big:
            for _ in range(num_nodes_repetition):
                a = torch.randn(mb, kb, device="cuda", dtype=torch.float16)
                b = torch.randn(kb, nb, device="cuda", dtype=torch.float16)
                c = torch.randn(mb, nb, device="cuda", dtype=torch.float16)
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
                    "threadblock_shape": [128, 256, 32],
                    "warp_count": [2, 4, 1],
                    "stages": 3,
                }
                # plan_tc.tile_description = {
                #     "threadblock_shape": [64, 128, 32],
                #     "warp_count": [2, 2, 1],
                #     "stages": 3,
                # }
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
                As.append(a)
                Bs.append(b)
                Cs.append(c)
                plan_tcs.append(plan_tc)
                arguments_tcs.append(arguments_tc)

        if do_small:
            for _ in range(num_nodes_repetition):
                a2 = torch.randn(m, k, device="cuda", dtype=torch.float16)
                b2 = torch.randn(k, n, device="cuda", dtype=torch.float16)
                c2 = torch.randn(m, n, device="cuda", dtype=torch.float16)
                plan_simt = cutlass.op.Gemm(
                    element=torch.float16,
                    layout=cutlass.LayoutType.RowMajor,
                    element_C=cutlass.DataType.void,
                    element_accumulator=cutlass.DataType.f16,
                )
                plan_simt.opclass = cutlass.OpcodeClass.TensorOp
                # for td in plan_simt.tile_descriptions():
                #     print(td)
                plan_simt.tile_description = {
                    "threadblock_shape": [128, 64, 32],
                    "warp_count": [2, 2, 1],
                    "stages": 3,
                }
                if do_big:
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
                A2s.append(a2)
                B2s.append(b2)
                C2s.append(c2)
                plan_simts.append(plan_simt)
                arguments_simts.append(arguments_simt)

        if do_big:
            for idx in range(num_nodes_repetition):
                constructor.capture_library_call_begin()
                plan_tcs[idx].operation.run(arguments_tcs[idx])
                constructor.capture_library_call_end()
        if do_small:
            if do_big:
                cm = constructor.registeredStreams[1].torch_stream
            else:  # null cm
                cm = contextlib.nullcontext()
            with cm:
                for idx in range(num_nodes_repetition):
                    constructor.capture_library_call_begin()
                    plan_simts[idx].operation.run(arguments_simts[idx])
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
        do_small=False, do_big=True
    )
    simt_time = test_canonical_procedure(do_small=True, do_big=False)  # SIMT
    print(
        "Non-interleaved time event: ",
        tensor_core_time,
        simt_time,
        tensor_core_time + simt_time,
    )
    print(
        "TFLOPs",
        num_nodes_repetition * gemm_tflops(mb, nb, kb, tensor_core_time),
        num_nodes_repetition * gemm_tflops(m, n, k, simt_time),
        num_nodes_repetition
        * (
            gemm_tflops(m, n, k, tensor_core_time + simt_time)
            + gemm_tflops(mb, nb, kb, tensor_core_time + simt_time)
        ),
    )

    interleaving_time = test_canonical_procedure(do_small=True, do_big=True)
    print(
        "Interleaved time event: ",
        interleaving_time,
    )
    print(
        "TFLOPs",
        num_nodes_repetition
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
    # print("Cutlass SIMT in parallel to TensorOp. Repeat with different data")
    # test_cutlass_simt_and_tensorop_interleave()
    print(
        "Triton SIMT in parallel Cutlass to TensorOp. Repeat with different"
        " data"
    )
    test_triton_simt_and_cutlass_tensorop_interleave()
    print(
        "Cutlass TensorOp big in parallel to small. Repeat with different data"
    )
    test_cutlass_tensorop_big_and_small_interleave()
