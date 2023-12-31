import intrasm_engine
import intrasm_engine_extensions as iex
import torch
import pycuda


# CUDAGraphConstructor + CUDAGraphCaptureNotifier
class TorchCUDAGraphConstructor:
    # torch.cuda.Stream(device=None)

    self.device: torch.device
    self.registeredPyCudaStreams: list[pycuda.driver.Stream]
    self.registeredStreams: list[torch.cuda.Stream]
    self.notifier: iex.CUDAGraphCaptureNotifier
    self.constructor: iex.CUDAExperimentalGraphConstructor
    # No need to store combined graph: proper GC is maintained by shared_ptr

    def __init__(self, device=torch.device("cuda")):
        self.device = device
        self.registeredPyCudaStreams = [
            intrasm_engine.current_pycuda_stream[self.device]
        ]
        self.registeredStreams = [intrasm_engine.current_stream[self.device]]
        self.notifier = iex.CUDAGraphCaptureNotifier()
        self.constructor = iex.CUDAExperimentalGraphConstructor(self.notifier)
        self.constructor.register_stream(self.registeredStreams[-1])

    def register_new_stream(self) -> torch.cuda.Stream:
        pycuda_stream = pycuda.driver.Stream()
        stream = torch.cuda.Stream(
            device=self.device, stream_ptr=pycuda_stream.handle
        )
        self.registeredPyCudaStreams.append(pycuda_stream)
        self.registeredStreams.append(stream)
        self.constructor.register_stream(self.registeredStreams[-1])
        return stream

    def register_new_streams(
        self, num_streams: int
    ) -> list[torch.cuda.Stream]:
        streams = []
        for idx in range(num_streams):
            streams.append(self.register_new_stream())
        return streams

    def capture_library_call_begin(self):
        self.notifier.capture_begin()
        assert torch.cuda.current_stream() in self.registeredStreams
        self.constructor.notify_before_invoking_library_call(
            torch.cuda.current_stream().cuda_stream
        )

    def capture_library_call_end(self):
        self.constructor.notify_after_invoking_library_call(
            torch.cuda.current_stream().cuda_stream
        )
        self.notifier.capture_end()

    def execute_graph(self):
        self.notifier.replay()
        self.constructor.execute_graph(self.registeredStreams[0])

    def synchronize(self):
        """Do device synchronize and destroy the graphExec"""
        torch.cuda.synchronize(device=self.device)
        self.constructor.destroy_graph_exec()

    # Context-related APIs. We are following the design of torch.cuda.Graph: https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/#api-example
