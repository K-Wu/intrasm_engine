"""FrameWork agnostic user-end APIs"""
import ctypes
import os
import platform
import subprocess
import sys
from .utils import *


def get_intrasm_path() -> str:
    """Find IntraSM Engine install path using pip"""

    command = [sys.executable, "-m", "pip", "show", "intrasm_engine"]
    result = subprocess.run(
        command, capture_output=True, check=True, text=True
    )
    result = result.stdout.replace("\n", ":").split(":")
    return result[result.index("Location") + 1].strip()


def _load_sputnik_library():
    """Load shared library with IntraSM Engine C framework-agnostic extensions"""

    system = platform.system()
    if system == "Linux":
        extension = "so"
    elif system == "Darwin":
        extension = "dylib"
    elif system == "Windows":
        extension = "dll"
    else:
        raise RuntimeError(f"Unsupported operating system ({system})")
    lib_name = "libsputnik." + extension
    dll_path = get_intrasm_path()
    dll_path = os.path.join(
        dll_path, "3rdparty", "sputnik", "build", "sputnik", lib_name
    )

    return ctypes.CDLL(dll_path, mode=ctypes.RTLD_GLOBAL)


def _get_cudart_path() -> str:
    import socket

    hostname = socket.gethostname()
    if hostname.startswith("hydro"):
        return "/projects/bbzc/kunwu2/spack/opt/spack/linux-rhel8-sandybridge/gcc-11.3.0/cuda-12.3.0-y3q7yk6kcuthrtopnxpjs4ui4knknetq/lib64/libcudart.so"
    elif hostname.startswith("kwu-csl227-99"):
        return "/usr/local/cuda-12/lib64/libcudart.so"
    else:
        raise RuntimeError(
            f"Unknown hostname ({hostname}) in .common.__init__"
        )


def _load_cudart_library() -> ctypes.CDLL:
    """Hack to get cudaStreamBeginCaptureToGraph while torch is prebuilt with CUDA 12.1"""
    cudart_path = _get_cudart_path()
    return ctypes.CDLL(cudart_path, mode=ctypes.RTLD_GLOBAL)


_CUDART_123_LIB_CTYPES = _load_cudart_library()

# _SPUTNIK_LIB_CTYPES = _load_sputnik_library()
