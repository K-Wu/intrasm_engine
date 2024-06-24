import torch
import pycuda
import pycuda.autoprimaryctx
import cupy


class CompoundStream:
    device: torch.device
    pycuda_stream: pycuda.driver.Stream
    torch_stream: torch.cuda.StreamContext
    cupy_stream: cupy.cuda.Stream

    def __init__(
        self,
        pycuda_stream: pycuda.driver.Stream,
        torch_stream: torch.cuda.StreamContext,
        cupy_stream: cupy.cuda.Stream,
    ):
        self.pycuda_stream = pycuda_stream
        self.torch_stream = torch_stream
        self.cupy_stream = cupy_stream
        self.device = self.torch_stream.stream.device

    # TODO: Incorporate this constructor
    def __init__alternative(
        self, pycuda_stream: pycuda.driver.Stream, device: torch.device
    ):
        self.pycuda_stream = pycuda_stream
        self.torch_stream = torch.cuda.stream(
            torch.cuda.Stream(
                stream_ptr=self.pycuda_stream.handle,
                device=device,
            )
        )
        self.cupy_stream = cupy.cuda.ExternalStream(
            self.torch_stream.stream.cuda_stream
        )
        self.device = self.torch_stream.stream.device

    def __enter__(self):
        """Both torch and cupy support setting the default stream, which is exactly what we do when CompounStream.__enter__() is called. In contrast, Pycuda does not have the default stream feature, and therefore we must set its stream explicitly on which the kernel launches to make sure the interleaving takes effect.
        The following is an example of the pycuda call trace. In this example, what we need to do is to pass the current stream to cutlass_utils.prepare_GemmArguments as the stream argument.
        Calling get_matmul_execs_cutlass_simt_f32 in test_cutlass_simt_f32_and_tensorop_interleave (https://github.com/K-Wu/intrasm_engine/blob/fd281674da5eff6b384cdd1b4e19678e232b58bb/tests/test_interleaving.py#L584)
          - Calling cutlass_utils.prepare_GemmArguments in _get_matmul_execs_cutlass_simt_f32 (https://github.com/K-Wu/intrasm_engine/blob/fd281674da5eff6b384cdd1b4e19678e232b58bb/tests/test_interleaving.py#L245)
        """
        print(
            "Warning: you need to set manually pycuda's stream in kernel"
            " launches."
        )
        # Reference: class StreamContext in https://github.com/pytorch/pytorch/blob/4bfaa6bc250f5ff5702703ea237f578a15bbe3b6/torch/cuda/__init__.py
        self.torch_stream.__enter__()
        # Reference: https://github.com/cupy/cupy/blob/2d02eaafed5b41ef148779eb34e2173fd5cf617e/cupy/cuda/stream.pyx
        self.cupy_stream.__enter__()
        return self

    def __exit__(self, *args):
        self.cupy_stream.__exit__(*args)
        self.torch_stream.__exit__(*args)
