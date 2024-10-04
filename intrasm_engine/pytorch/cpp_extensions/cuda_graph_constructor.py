import intrasm_engine
import intrasm_engine_extensions as iex
import torch
from torch import nn
import pycuda
import pycuda.autoprimaryctx
from ...common.compound_streams import CompoundStream
import cupy


# CUDAGraphConstructor + CUDAGraphCaptureNotifier
class TorchCUDAGraphConstructor:
    # torch.cuda.Stream(device=None)

    device: torch.device
    # registeredPyCudaStreams: list[pycuda.driver.Stream]
    # registeredStreams: list[torch.cuda.StreamContext]
    registeredStreams: list[CompoundStream]
    notifier: iex.CUDAGraphCaptureNotifier
    constructor: iex.CUDAExperimentalGraphConstructor
    # No need to store combined graph: proper GC is maintained by shared_ptr

    def __init__(
        self, device=torch.device(f"cuda:{torch.cuda.current_device()}")
    ):
        self.device = device
        # self.registeredPyCudaStreams = [
        #     intrasm_engine.current_pycuda_stream[self.device]
        # ]
        self.registeredStreams = [intrasm_engine.current_stream[self.device]]
        self.notifier = iex.CUDAGraphCaptureNotifier()
        self.constructor = iex.CUDAExperimentalGraphConstructor()
        self.constructor.register_stream(
            self.registeredStreams[0].torch_stream.stream.cuda_stream
        )

    def get_primary_stream(self) -> CompoundStream:
        """the first stream is the stream where streams join and the graph is executed."""
        return self.registeredStreams[0]

    def register_new_stream(self) -> CompoundStream:
        pycuda_stream = pycuda.driver.Stream()
        torch_stream = torch.cuda.stream(
            torch.cuda.Stream(
                device=self.device, stream_ptr=pycuda_stream.handle
            )
        )
        cupy_stream = cupy.cuda.ExternalStream(torch_stream.stream.cuda_stream)
        stream = CompoundStream(pycuda_stream, torch_stream, cupy_stream)
        self.registeredStreams.append(stream)
        self.constructor.register_stream(
            self.registeredStreams[-1].torch_stream.stream.cuda_stream
        )
        return stream

    def print_graph(self):
        self.constructor.print_graph()

    def register_new_streams(self, num_streams: int) -> list[CompoundStream]:
        streams = []
        for idx in range(num_streams):
            streams.append(self.register_new_stream())
        return streams

    def assert_current_stream_registered(self):
        assert torch.cuda.current_stream() in {
            context.torch_stream.stream for context in self.registeredStreams
        }, "Current stream is not registered."

    def add_event_record_node(
        self, event: torch.cuda.Event, torch_stream: torch.cuda.StreamContext
    ):
        # PyTorch's event is only created after event.record() is called. Before that, event.cuda_event is 0.
        assert event.cuda_event != 0, (
            "Event uninitialized! PyTorch initialize the event lazily. Please"
            " call event.record() after creating the event object and before"
            " the real recording to make sure the event is initialized."
        )

        self.constructor.add_event_record_node(
            event.cuda_event, torch_stream.stream.cuda_stream
        )

    def add_stream_wait_event_node(
        self, torch_stream: torch.cuda.StreamContext, event: torch.cuda.Event
    ):
        # PyTorch's event is only created after event.record() is called. Before that, event.cuda_event is 0.
        assert event.cuda_event != 0, (
            "Event uninitialized! PyTorch initialize the event lazily. Please"
            " call event.record() after creating the event object and before"
            " the real recording to make sure the event is initialized."
        )

        self.constructor.add_stream_wait_event_node(
            torch_stream.stream.cuda_stream, event.cuda_event
        )

    def capture_library_call_begin(self):
        self.assert_current_stream_registered()
        self.notifier.capture_begin(None)
        self.constructor.notify_before_invoking_library_call(
            torch.cuda.current_stream().cuda_stream
        )
        self.notifier.assert_capture_has_begun()

    def capture_library_call_end(self):
        self.constructor.notify_after_invoking_library_call(
            torch.cuda.current_stream().cuda_stream
        )
        self.notifier.capture_end()

    def instantiate_graph_exec(self):
        self.constructor.instantiate_graph_exec()

    def execute_graph(self):
        self.notifier.replay()  # TODO: check if this line needs to be before instantiate_graph()
        # Use the first stream to execute the graph
        self.constructor.execute_graph(
            self.registeredStreams[0].torch_stream.stream.cuda_stream
        )

    def join(self, streams: list[CompoundStream], dest_stream: CompoundStream):
        self.constructor.join(
            [stream.torch_stream.stream.cuda_stream for stream in streams],
            dest_stream.torch_stream.stream.cuda_stream,
        )

    def synchronize(self):
        """Do device synchronize and destroy the graphExec"""
        torch.cuda.synchronize(device=self.device)

    def destroy_graph_exec(self):
        self.constructor.destroy_graph_exec()

    # Context-related APIs. We are following the design of torch.cuda.Graph: https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/#api-example
    def __enter__(self):
        """
        Usage:
        with torch_stream_context:
            with cudagraph_constructor:
                # Do stuff
        """
        self.capture_library_call_begin()
        return self

    def __exit__(self, *args):
        self.capture_library_call_end()
