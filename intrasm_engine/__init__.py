"""Top level package"""
from . import common

try:
    from . import pytorch
except ImportError as e:
    pass

import torch
import pycuda
import pycuda.autoprimaryctx
import sys

# This is a pointer to the module object instance itself.
# Reference: https://stackoverflow.com/a/35904211/5555077
this = sys.modules[__name__]
this.current_pycuda_stream: dict[torch.device, pycuda.driver.Stream]
this.current_stream: dict[torch.device, torch.cuda.Stream]


def initialize_current_streams():
    """
    The canonical way to refer to these current streams objects are
    import sparta
    sparta.current_stream
    Reference: https://docs.python.org/3/faq/programming.html#how-do-i-share-global-variables-across-modules
    Reference on how python caches modules so that they are only loaded once: https://docs.python.org/3/library/sys.html#sys.modules
    """
    num_gpus = torch.cuda.device_count()
    this.current_pycuda_stream = {}
    this.current_stream = {}
    for i in range(num_gpus):
        this.current_pycuda_stream[
            torch.device(f"cuda:{i}")
        ] = pycuda.driver.Stream()
        this.current_stream[torch.device(f"cuda:{i}")] = torch.cuda.Stream(
            stream_ptr=this.current_pycuda_stream[
                torch.device(f"cuda:{i}")
            ].handle
        )

    torch.cuda.set_device(0)
    torch.cuda.set_stream(this.current_stream[torch.device(f"cuda:{0}")])

    ## Triton
    # Triton ops uses its own runtime.driver get_current_stream, underlain by torch._C._cuda_getCurrentRawStream, to identify the stream to run the kernel on. Reference: https://github.com/openai/triton/blob/a767ca41e189988740d35cbb9aecd873c4874a62/python/triton/runtime/jit.py#L340
    # https://github.com/openai/triton/blob/a767ca41e189988740d35cbb9aecd873c4874a62/python/triton/runtime/driver.py#L37

    ## Cupy
    # To set cupy stream during cupy computation, use cupy.cuda.ExternalStream. Reference: https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.ExternalStream.html#cupy-cuda-externalstream
    # An example of setting up cupy-torch interoperability: https://github.com/cupy/cupy/blob/8368780c911b7a7fb7b881ec57ac4f53732c083f/docs/source/user_guide/interoperability.rst#cuda-stream-pointers


initialize_current_streams()
