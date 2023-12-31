from cupy.cuda import device as _device

# Reference: https://github.com/cupy/cupy/blob/2d02eaafed5b41ef148779eb34e2173fd5cf617e/cupy/cuda/device.pyx
# Example: https://github.com/cupy/cupy/blob/2d02eaafed5b41ef148779eb34e2173fd5cf617e/cupyx/cusparse.py
h = _device.get_cusparse_handle()

# Reference: https://github.com/cupy/cupy/blob/2d02eaafed5b41ef148779eb34e2173fd5cf617e/cupy/cuda/device.pyx
# https://github.com/cupy/cupy/blob/2d02eaafed5b41ef148779eb34e2173fd5cf617e/cupy_backends/cuda/libs/cusparse.pyx
from cupy_backends.cuda.libs import cusparse

h2 = cusparse.create()
cusparse.setStream(h2, 0)
cusparse.setStream(h, 0)
sid = cusparse.getStream(h)
sid2 = cusparse.getStream(h2)
cusparse.setStream(h, 0)
cusparse.setStream(h2, 0)

import torch
import cupy

ts = torch.cuda.Stream()
cs = cupy.cuda.ExternalStream(ts.cuda_stream)
print(
    "ptr value retrival from Pytorch Stream and cupy Stream",
    ts.cuda_stream,
    cs.ptr,
)
# The `stream` variable in setStream argument and getStream return values are both the stream.ptr value
cusparse.setStream(h2, cs.ptr)
