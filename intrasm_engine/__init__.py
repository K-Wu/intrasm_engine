"""Top level package"""
from .common import compound_streams


import torch
import pycuda
import pycuda.autoprimaryctx
import cupy
import sys

# This is a pointer to the module object instance itself.
# Reference: https://stackoverflow.com/a/35904211/5555077
this = sys.modules[__name__]
this.current_stream: dict[torch.device, compound_streams.CompoundStream]


def initialize_current_streams():
    """
    The canonical way to refer to these current streams objects are
    import sparta
    sparta.current_stream
    Reference: https://docs.python.org/3/faq/programming.html#how-do-i-share-global-variables-across-modules
    Reference on how python caches modules so that they are only loaded once: https://docs.python.org/3/library/sys.html#sys.modules
    """
    # PyCUDA stream -> Torch stream
    # Torch stream -> Cupy stream. check reference listed in test/test_cupy_library_handles.py
    num_gpus = torch.cuda.device_count()
    this.current_stream = {}
    for i in range(num_gpus):
        current_pycuda_stream = pycuda.driver.Stream()
        current_torch_stream = torch.cuda.stream(
            torch.cuda.Stream(
                stream_ptr=current_pycuda_stream.handle,
                device=f"cuda:{i}",
            )
        )
        current_cupy_stream = cupy.cuda.ExternalStream(
            current_torch_stream.stream.cuda_stream
        )
        this.current_stream[
            torch.device(f"cuda:{i}")
        ] = compound_streams.CompoundStream(
            current_pycuda_stream,
            current_torch_stream,
            current_cupy_stream,
        )

    torch.cuda.set_device(0)
    torch.cuda.set_stream(
        this.current_stream[torch.device(f"cuda:{0}")].torch_stream.stream
    )

    ## Triton uses Torch stream.
    # Triton ops uses its own runtime.driver get_current_stream, underlain by torch._C._cuda_getCurrentRawStream, to identify the stream to run the kernel on. Reference: https://github.com/openai/triton/blob/a767ca41e189988740d35cbb9aecd873c4874a62/python/triton/runtime/jit.py#L340
    # https://github.com/openai/triton/blob/a767ca41e189988740d35cbb9aecd873c4874a62/python/triton/runtime/driver.py#L37


initialize_current_streams()
print(
    (
        "Print handle to force cublas initialization (otherwise first matmul"
        " captured in the graph may fail):"
    ),
    torch.cuda.current_blas_handle(),
)

from . import common
from . import pytorch
from .pytorch import utils as pytorch_utils
import sys
import os

dir_of_this_script = os.path.dirname(sys.modules[__name__].__file__)
# Try the following if the above does not work
# dir_of_this_script = os.path.dirname(os.path.realpath(__file__))
torch.classes.load_library(
    os.path.join(
        dir_of_this_script,
        "..",
        f"intrasm_engine_extensions.cpython-{sys.version_info.major}{sys.version_info.minor}-x86_64-linux-gnu.so",
    )
)

# pytorch_utils.set_tf32_use_tensor_core()
pytorch_utils.set_float_16_reduced_precision()
