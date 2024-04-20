# From https://github.com/NVIDIA/TransformerEngine/blob/e7261e116d3a27c333b8d8e3972dbea20288b101/setup.py
import setuptools
from pathlib import Path
from typing import List, Optional, Union, Tuple
import os
import sys
from functools import lru_cache
import subprocess
from subprocess import CalledProcessError
from setuptools.command.build_ext import build_ext
import shutil
import tempfile
import re
import ctypes


# Project directory root
root_path: Path = Path(__file__).resolve().parent


def frameworks() -> List[str]:
    return ["pytorch"]  # Only support PyTorch for now


# Call once in global scope since in the future this function may manipulate the
# command-line arguments. Future calls will use a cached value.
frameworks()


@lru_cache(maxsize=1)
def get_intrasm_engine_version() -> str:
    """Transformer Engine version string

    Includes Git commit as local version, unless suppressed with
    MYIE_NO_LOCAL_VERSION environment variable.

    """
    with open(root_path / "VERSION", "r") as f:
        version = f.readline().strip()
    if not int(os.getenv("MYIE_NO_LOCAL_VERSION", "0")):
        try:
            output = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                cwd=root_path,
                check=True,
                universal_newlines=True,
            )
        except (CalledProcessError, OSError):
            pass
        else:
            commit = output.stdout.strip()
            version += f"+{commit}"
    return version


@lru_cache(maxsize=1)
def with_debug_build() -> bool:
    """Whether to build with a debug configuration"""
    for arg in sys.argv:
        if arg == "--debug":
            sys.argv.remove(arg)
            return True
    if int(os.getenv("NVTE_BUILD_DEBUG", "0")):
        return True
    return False


# Call once in global scope since this function manipulates the
# command-line arguments. Future calls will use a cached value.
with_debug_build()


def found_cmake() -> bool:
    """ "Check if valid CMake is available

    CMake 3.18 or newer is required.

    """

    # Check if CMake is available
    try:
        _cmake_bin = cmake_bin()
    except FileNotFoundError:
        return False

    # Query CMake for version info
    output = subprocess.run(
        [_cmake_bin, "--version"],
        capture_output=True,
        check=True,
        universal_newlines=True,
    )
    match = re.search(r"version\s*([\d.]+)", output.stdout)
    version = match.group(1).split(".")
    version = tuple(int(v) for v in version)
    return version >= (3, 18)


def cmake_bin() -> Path:
    """Get CMake executable

    Throws FileNotFoundError if not found.

    """

    # Search in CMake Python package
    _cmake_bin: Optional[Path] = None
    try:
        import cmake
    except ImportError:
        pass
    else:
        cmake_dir = Path(cmake.__file__).resolve().parent
        _cmake_bin = cmake_dir / "data" / "bin" / "cmake"
        if not _cmake_bin.is_file():
            _cmake_bin = None

    # Search in path
    if _cmake_bin is None:
        _cmake_bin = shutil.which("cmake")
        if _cmake_bin is not None:
            _cmake_bin = Path(_cmake_bin).resolve()

    # Return executable if found
    if _cmake_bin is None:
        raise FileNotFoundError("Could not find CMake executable")
    return _cmake_bin


def found_ninja() -> bool:
    """ "Check if Ninja is available"""
    return shutil.which("ninja") is not None


def found_pybind11() -> bool:
    """ "Check if pybind11 is available"""

    # Check if Python package is installed
    try:
        import pybind11
    except ImportError:
        pass
    else:
        return True

    # Check if CMake can find pybind11
    if not found_cmake():
        return False
    try:
        subprocess.run(
            [
                "cmake",
                "--find-package",
                "-DMODE=EXIST",
                "-DNAME=pybind11",
                "-DCOMPILER_ID=CXX",
                "-DLANGUAGE=CXX",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (CalledProcessError, OSError):
        pass
    else:
        return True
    return False


def cuda_version() -> Tuple[int, ...]:
    """CUDA Toolkit version as a (major, minor) tuple

    Throws FileNotFoundError if NVCC is not found.

    """

    # Try finding NVCC
    nvcc_bin: Optional[Path] = None
    if nvcc_bin is None and os.getenv("CUDA_HOME"):
        # Check in CUDA_HOME
        cuda_home = Path(os.getenv("CUDA_HOME"))
        nvcc_bin = cuda_home / "bin" / "nvcc"
    if nvcc_bin is None:
        # Check if nvcc is in path
        nvcc_bin = shutil.which("nvcc")
        if nvcc_bin is not None:
            nvcc_bin = Path(nvcc_bin)
    if nvcc_bin is None:
        # Last-ditch guess in /usr/local/cuda
        cuda_home = Path("/usr/local/cuda")
        nvcc_bin = cuda_home / "bin" / "nvcc"
    if not nvcc_bin.is_file():
        raise FileNotFoundError(f"Could not find NVCC at {nvcc_bin}")

    # Query NVCC for version info
    output = subprocess.run(
        [nvcc_bin, "-V"],
        capture_output=True,
        check=True,
        universal_newlines=True,
    )
    match = re.search(r"release\s*([\d.]+)", output.stdout)
    version = match.group(1).split(".")
    return tuple(int(v) for v in version)


def setup_requirements() -> Tuple[List[str], List[str], List[str]]:
    """Setup Python dependencies

    Returns dependencies for build, runtime, and testing.

    """

    # Common requirements
    setup_reqs: List[str] = []
    install_reqs: List[str] = [
        "pydantic",
        "GitPython",
        "cupy",
        "cuda-python",
        "pycuda",
        "nvidia-cutlass",
        # Suppress it to avoid frequent update leading to costly recompilation
        # (
        #     "xformers @"
        #     " git+https://github.com/facebookresearch/xformers.git@main#egg=xformers"
        # ),
        (
            "torchknickknacks @"
            " git+https://github.com/AlGoulas/torchknickknacks.git"
        ),
        "triton_autotuning @ git+https://github.com/K-Wu/triton_autotuning",
        # TODO: enable the following and test if it works
        # "SparTA @ git+ssh://git@github.com:K-Wu/SparTA@main",
    ]
    test_reqs: List[str] = [
        "pytest",
    ]

    def add_unique(l: List[str], vals: Union[str, List[str]]) -> None:
        """Add entry to list if not already included"""
        if isinstance(vals, str):
            vals = [vals]
        for val in vals:
            if val not in l:
                l.append(val)

    # Requirements that may be installed outside of Python
    if not found_cmake():
        add_unique(setup_reqs, "cmake>=3.18")
    if not found_ninja():
        add_unique(setup_reqs, "ninja")

    # Framework-specific requirements
    if "pytorch" in frameworks():
        add_unique(
            install_reqs,
            ["torch", "flash-attn>=1.0.6,<=2.3.3,!=2.0.9,!=2.1.0"],
        )
        add_unique(test_reqs, ["numpy", "onnxruntime", "torchvision"])
    if "jax" in frameworks():
        if not found_pybind11():
            add_unique(setup_reqs, "pybind11")
        add_unique(install_reqs, ["jax", "flax>=0.7.1"])
        add_unique(test_reqs, ["numpy", "praxis"])
    if "paddle" in frameworks():
        add_unique(install_reqs, "paddlepaddle-gpu")
        add_unique(test_reqs, "numpy")

    return setup_reqs, install_reqs, test_reqs


class CMakeExtension(setuptools.Extension):
    """CMake extension module"""

    def __init__(
        self,
        name: str,
        cmake_path: Path,
        cmake_flags: Optional[List[str]] = None,
    ) -> None:
        super().__init__(name, sources=[])  # No work for base class
        self.cmake_path: Path = cmake_path
        self.cmake_flags: List[str] = (
            [] if cmake_flags is None else cmake_flags
        )

    def _build_cmake(self, build_dir: Path, install_dir: Path) -> None:
        # Make sure paths are str
        _cmake_bin = str(cmake_bin())
        cmake_path = str(self.cmake_path)
        build_dir = str(build_dir)
        install_dir = str(install_dir)

        # CMake configure command
        build_type = "Debug" if with_debug_build() else "Release"
        configure_command = [
            _cmake_bin,
            "-S",
            cmake_path,
            "-B",
            build_dir,
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        ]
        configure_command += self.cmake_flags
        if found_ninja():
            configure_command.append("-GNinja")
        try:
            import pybind11
        except ImportError:
            pass
        else:
            pybind11_dir = Path(pybind11.__file__).resolve().parent
            pybind11_dir = pybind11_dir / "share" / "cmake" / "pybind11"
            configure_command.append(f"-Dpybind11_DIR={pybind11_dir}")

        # CMake build and install commands
        build_command = [_cmake_bin, "--build", build_dir]
        install_command = [_cmake_bin, "--install", build_dir]

        # Run CMake commands
        for command in [configure_command, build_command, install_command]:
            print(f"Running command {' '.join(command)}")
            try:
                subprocess.run(command, cwd=build_dir, check=True)
            except (CalledProcessError, OSError) as e:
                raise RuntimeError(f"Error when running CMake: {e}")


# PyTorch extension modules require special handling
if "pytorch" in frameworks():
    from torch.utils.cpp_extension import BuildExtension
elif "paddle" in frameworks():
    from paddle.utils.cpp_extension import BuildExtension
else:
    from setuptools.command.build_ext import build_ext as BuildExtension


class CMakeBuildExtension(BuildExtension):
    """Setuptools command with support for CMake extension modules"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def run(self) -> None:
        # Build CMake extensions
        for ext in self.extensions:
            if isinstance(ext, CMakeExtension):
                print(f"Building CMake extension {ext.name}")
                with tempfile.TemporaryDirectory() as build_dir:
                    build_dir = Path(build_dir)
                    package_path = Path(self.get_ext_fullpath(ext.name))
                    install_dir = package_path.resolve().parent
                    ext._build_cmake(
                        build_dir=build_dir,
                        install_dir=install_dir,
                    )

        # Paddle requires linker search path for libintrasm_engine.so
        paddle_ext = None
        if "paddle" in frameworks():
            for ext in self.extensions:
                if "paddle" in ext.name:
                    ext.library_dirs.append(self.build_lib)
                    paddle_ext = ext
                    break

        # Build non-CMake extensions as usual
        all_extensions = self.extensions
        self.extensions = [
            ext
            for ext in self.extensions
            if not isinstance(ext, CMakeExtension)
        ]
        super().run()
        self.extensions = all_extensions

        # Manually write stub file for Paddle extension
        if paddle_ext is not None:
            # Load libintrasm_engine.so to avoid linker errors
            for path in Path(self.build_lib).iterdir():
                if path.name.startswith("libintrasm_engine."):
                    ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)

            # Figure out stub file path
            module_name = paddle_ext.name
            assert module_name.endswith(
                "_pd_"
            ), "Expected Paddle extension module to end with '_pd_'"
            stub_name = module_name[:-4]  # remove '_pd_'
            stub_path = os.path.join(self.build_lib, stub_name + ".py")

            # Figure out library name
            # Note: This library doesn't actually exist. Paddle
            # internally reinserts the '_pd_' suffix.
            so_path = self.get_ext_fullpath(module_name)
            _, so_ext = os.path.splitext(so_path)
            lib_name = stub_name + so_ext

            # Write stub file
            print(f"Writing Paddle stub for {lib_name} into file {stub_path}")
            from paddle.utils.cpp_extension.extension_utils import (
                custom_write_stub,
            )


def setup_common_extension() -> CMakeExtension:
    """Setup CMake extension for common library

    Also builds JAX or userbuffers support if needed.

    """
    cmake_flags = []
    return CMakeExtension(
        name="intrasm_engine",
        cmake_path=root_path / "intrasm_engine",
        cmake_flags=cmake_flags,
    )


def _all_files_in_dir(path: Path) -> List[Path]:
    return list(path.iterdir())


def _all_files_with_suffix_in_dir(path: Path, suffix: str) -> List[Path]:
    filenames = _all_files_in_dir(path)
    return [filename for filename in filenames if filename.suffix == suffix]


def _all_cpp_files_in_dir(path: Path) -> List[Path]:
    return _all_files_with_suffix_in_dir(path, ".cpp")


def _all_cu_files_in_dir(path: Path) -> List[Path]:
    return _all_files_with_suffix_in_dir(path, ".cu")


def setup_pytorch_extension() -> setuptools.Extension:
    """Setup CUDA extension for PyTorch support"""

    # Source files
    src_dir: Path = root_path / "intrasm_engine" / "pytorch" / "csrc"
    extensions_dir: Path = src_dir / "extensions"
    sources = (
        [
            src_dir / "MyStackClass.cpp",
            # src_dir / "common.cu",
            # src_dir / "ts_fp8_op.cpp",
        ]
        + _all_cpp_files_in_dir(extensions_dir)
        + _all_cu_files_in_dir(extensions_dir)
    )

    # Header files
    include_dirs = [
        root_path / "intrasm_engine" / "common" / "include",
        root_path / "intrasm_engine" / "pytorch" / "csrc",
        root_path / "intrasm_engine",
        root_path / "3rdparty" / "sputnik",
        # root_path / "3rdparty" / "cudnn-frontend" / "include",
    ]

    # Compiler flags
    cxx_flags = ["-O3"]
    nvcc_flags = [
        "-O3",
        "-gencode",
        "arch=compute_70,code=sm_70",
        "-gencode",
        "arch=compute_86,code=sm_86",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ]

    # Version-dependent CUDA options
    try:
        version = cuda_version()
    except FileNotFoundError:
        print("Could not determine CUDA Toolkit version")
    else:
        if version >= (11, 2):
            nvcc_flags.extend(["--threads", "4"])
        if version >= (11, 0):
            nvcc_flags.extend(["-gencode", "arch=compute_80,code=sm_80"])
        if version >= (11, 8):
            nvcc_flags.extend(["-gencode", "arch=compute_90,code=sm_90"])

    # Construct PyTorch CUDA extension
    sources = [str(path) for path in sources]
    include_dirs = [str(path) for path in include_dirs]
    from torch.utils.cpp_extension import CUDAExtension

    return CUDAExtension(
        name="intrasm_engine_extensions",
        sources=sources,
        include_dirs=include_dirs,
        # libraries=["intrasm_engine"], ### TODO (tmoon) Debug linker errors
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
    )


def main():
    import os

    if "MAX_JOBS" not in os.environ:
        print(
            "WARNING: MAX_JOBS is not set, defaulting to 4. This is vital to"
            " ensure that the dependency xformer won't exaust the memory of"
            " your machine during installation."
        )
        os.environ["MAX_JOBS"] = "4"

    # Submodules to install
    packages = setuptools.find_packages(
        include=["intrasm_engine", "intrasm_engine.*"],
    )

    # Dependencies
    setup_requires, install_requires, test_requires = setup_requirements()

    # COnfigure package
    setuptools.setup(
        name="intrasm_engine",
        version=get_intrasm_engine_version(),
        description="Intra-SM Parallelism Engine via CUDA Graph",
        packages=packages,
        ext_modules=[setup_common_extension(), setup_pytorch_extension()],
        cmdclass={"build_ext": CMakeBuildExtension},
        setup_requires=setup_requires,
        install_requires=install_requires,
        extras_require={"test": test_requires},
        license_files=("LICENSE",),
    )


if __name__ == "__main__":
    main()
