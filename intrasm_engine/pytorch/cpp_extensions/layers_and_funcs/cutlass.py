import sys
import torch
from ....common import external_kernels  # sputnik and cutlass

this = sys.modules[__name__]
this.single_kernel_autograd_functions: list[type[torch.autograd.Function]] = []
