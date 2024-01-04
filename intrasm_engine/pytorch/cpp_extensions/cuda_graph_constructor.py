import intrasm_engine
import intrasm_engine_extensions as iex
import torch
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

    def register_new_streams(self, num_streams: int) -> list[CompoundStream]:
        streams = []
        for idx in range(num_streams):
            streams.append(self.register_new_stream())
        return streams

    def assert_current_stream_registered(self):
        assert torch.cuda.current_stream() in {
            context.torch_stream.stream for context in self.registeredStreams
        }, "Current stream is not registered."

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

    def execute_graph(self):
        self.notifier.replay()
        # Use the first stream to execute the graph
        self.constructor.execute_graph(
            self.registeredStreams[0].torch_stream.stream.cuda_stream
        )

    def synchronize(self):
        """Do device synchronize and destroy the graphExec"""
        torch.cuda.synchronize(device=self.device)
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


class CUDAGraphModulePreviousLayerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *input):
        """
        torch.autograd.Function's forward function has **kwargs to parameterize the forward+backward function pair. We created CUDAGraphModulePreviousLayerFunction to replay the graph captured in the constructor.
        So we only need to pass all the tensors as is to make sure the replay does executed by the PyTorch autograd engine. We do not need to pass **kwargs from the replayed function.
        In future, if we need to parameterize the CUDAGraphModulePreviousLayerFunction itself, we may add **kwargs.
        """
        ctx.backward_constructor = input[0]
        ctx.save_for_backward(*input[1:])
        tensor_input = input[1:]
        return (*tensor_input,)

    @staticmethod
    def backward(ctx, *grad_tensor_input):
        backward_constructor = ctx.backward_constructor
        backward_constructor.execute_graph()
        backward_constructor.synchronize()
        return (None, *grad_tensor_input)


class CUDAGraphModuleNextLayerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *input):
        """
        torch.autograd.Function's forward function has **kwargs to parameterize the forward+backward function pair. We created CUDAGraphModuleNextLayerFunction to replay the graph captured in the constructor.
        So we only need to pass all the tensors as is to make sure the replay does executed by the PyTorch autograd engine. We do not need to pass **kwargs from the replayed function.
        In future, if we need to parameterize the CUDAGraphModuleNextLayerFunction itself, we may add **kwargs.
        """
        forward_constructor = input[0]
        ctx.forward_constructor = forward_constructor
        tensor_input = input[1:]
        ctx.save_for_backward(*tensor_input)
        forward_constructor.execute_graph()
        forward_constructor.synchronize()
        return (*tensor_input,)

    @staticmethod
    def backward(ctx, *grad_tensor_input):
        return (None, *grad_tensor_input)


class CUDAGraphModulePreviousLayer(torch.nn.Module):
    def __init__(self, backward_constructor: TorchCUDAGraphConstructor):
        super().__init__()
        self.backward_constructor = backward_constructor

    def forward(self, *input):
        return CUDAGraphModulePreviousLayerFunction.apply(
            self.backward_constructor, *input
        )


class CUDAGraphModuleNextLayer(torch.nn.Module):
    def __init__(self, forward_constructor: TorchCUDAGraphConstructor):
        super().__init__()
        self.forward_constructor = forward_constructor

    def forward(self, *input):
        return CUDAGraphModuleNextLayerFunction.apply(
            self.forward_constructor, *input
        )
