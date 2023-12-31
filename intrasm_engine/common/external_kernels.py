"""This module accomodates loaded sputnik and cutlass cubins."""
import pycuda.autoprimaryctx  # Initialize pycuda
import pycuda.driver

from . import cubin_loading_utils
import sys

this = sys.modules[__name__]


this.CUBIN_MODULES: dict[str, pycuda.driver.Module] = {}


def load_cubin(cubin_name: str):
    """Load the cubin file, which requires building sputnik with cubin enabled by configuring cmake with -DBUILD_CUBIN=ON"""
    if cubin_name not in CUBIN_MODULES:
        this.CUBIN_MODULES[cubin_name] = pycuda.driver.module_from_file(
            cubin_loading_utils.CUBIN_PATHS[cubin_name]
        )


# TODO: add support to cutlass kernels
# nvidia-cutlass is the official cutlass python package. It supports torch tensors, JIT compilation, and JIT compilation cache.
# JIT and Torch AOT example: https://github.com/NVIDIA/cutlass/blob/8098336d512ef089a2f0e0fa172d5ff5cb18eca5/examples/python/02_pytorch_extension_grouped_gemm.ipynb
