import intrasm_engine
import intrasm_engine_extensions as iex  # Loading iex along without intrasm_engine will trigger libc10.so not found error
import torch
from intrasm_engine.pytorch.cpp_extensions.cuda_graph_constructor import (
    TorchCUDAGraphConstructor,
)
import time
from intrasm_engine.common import cutlass_utils
from cuda import cuda


def test_replay_cutlass_gemm_with_events():
    import cutlass
    import random

    dtype = torch.float16
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

    constructor = TorchCUDAGraphConstructor()
    constructor.register_new_stream()

    plan = cutlass.op.Gemm(
        element=dtype,
        layout=cutlass.LayoutType.RowMajor,
        element_C=cutlass.DataType.void,
        element_accumulator=cutlass.DataType.f16,
    )
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
            init_value=constructor.registeredStreams[
                0
            ].torch_stream.stream.cuda_stream
        ),
    )

    plan2 = cutlass.op.Gemm(
        element=dtype,
        layout=cutlass.LayoutType.RowMajor,
        element_C=cutlass.DataType.void,
        element_accumulator=cutlass.DataType.f16,
    )
    (
        A,
        B,
        D,
    ) = generate_problems()
    arguments2 = cutlass_utils.prepare_GemmArguments(
        plan2,
        A,
        B,
        None,
        D,
        print_module=False,
        stream=cuda.CUstream(
            init_value=constructor.registeredStreams[
                1
            ].torch_stream.stream.cuda_stream
        ),
    )

    plans = [plan, plan2]
    arguments_lists = [arguments, arguments2]

    start_event = torch.cuda.Event(enable_timing=True)
    start_event.record(
        stream=constructor.registeredStreams[0].torch_stream.stream
    )
    start_event2 = torch.cuda.Event(enable_timing=True)
    start_event2.record(
        stream=constructor.registeredStreams[1].torch_stream.stream
    )
    start_events = [start_event, start_event2]
    end_event = torch.cuda.Event(enable_timing=True)
    end_event.record(
        stream=constructor.registeredStreams[0].torch_stream.stream
    )
    time.sleep(5)
    end_event2 = torch.cuda.Event(enable_timing=True)
    end_event2.record(
        stream=constructor.registeredStreams[1].torch_stream.stream
    )
    end_events = [end_event, end_event2]

    join_event = torch.cuda.Event(enable_timing=True)
    join_event.record(
        stream=constructor.registeredStreams[1].torch_stream.stream
    )

    for idx in range(2):
        with constructor.registeredStreams[idx].torch_stream as cm:
            constructor.add_event_record_node(
                start_events[idx],
                constructor.registeredStreams[idx].torch_stream,
            )
            constructor.capture_library_call_begin()
            plans[idx].operation.run(arguments_lists[idx])
            constructor.capture_library_call_end()
            constructor.add_event_record_node(
                end_events[idx],
                constructor.registeredStreams[idx].torch_stream,
            )
            if idx == 1:
                constructor.add_event_record_node(
                    join_event, constructor.registeredStreams[1].torch_stream
                )

    constructor.add_stream_wait_event_node(
        constructor.registeredStreams[0].torch_stream, join_event
    )

    constructor.instantiate_graph_exec()
    constructor.execute_graph()
    constructor.synchronize()
    constructor.destroy_graph_exec()

    print("elapsed_time: ", start_event.elapsed_time(end_event))
    print("elapsed_time: ", start_event2.elapsed_time(end_event2))
    print("elapsed_time: ", start_event.elapsed_time(end_event2))
    print("elapsed_time: ", start_event2.elapsed_time(join_event))
    print("elapsed_time: ", start_event.elapsed_time(join_event))

    D_torch = A @ B

    assert torch.allclose(D, D_torch)


if __name__ == "__main__":
    test_replay_cutlass_gemm_with_events()
