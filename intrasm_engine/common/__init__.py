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


# Hack to get cudaStreamBeginCaptureToGraph while torch is prebuilt with CUDA 12.1
_CUDART_123_LIB_CTYPES = ctypes.CDLL(
    "/usr/local/cuda-12/lib64/libcudart.so", mode=ctypes.RTLD_GLOBAL
)

# _SPUTNIK_LIB_CTYPES = _load_sputnik_library()
