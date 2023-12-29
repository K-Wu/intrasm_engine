import subprocess
import os
from .utils import get_git_root_recursively


# From /HET/hrt/utils/stat_sass_inst.py
def demangle_cuda_function_name(func_name: str) -> str:
    """Demangle CUDA function name."""
    return (
        subprocess.check_output(["cu++filt", func_name])
        .decode("utf-8")
        .strip()
    )
    return


def refresh_cuda_symbol_table(symbol_table_filename: str, cubin_filename: str):
    symbol_table_filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        symbol_table_filename,
    )
    # Use `cuobjdump -symbols`` to get the symbol table
    subprocess.check_call(
        [
            "cuobjdump",
            "-symbols",
            cubin_filename,
        ],
        stdout=open(symbol_table_filename, "w"),
    )
    return


def get_signature_to_mangle_name_map(
    symbol_table_filename: str,
) -> dict[str, str]:
    """This function returns a dictionary with key being the signature of the function, and value being the mangled name of the function.
    This is non-trivial because mangle name contains certain random strings, e.g., when dealing with unnamed namespace.
    """
    results: dict[str, str] = {}
    # Read the symbol table .txt at the same directory as this script
    with open(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            symbol_table_filename,
        )
    ) as fd:
        for line in fd:
            line = line.strip()
            if line.startswith("STT_FUNC"):
                mangled_func_name = line.split(" ")[-1]
                results[
                    demangle_cuda_function_name(mangled_func_name)
                ] = mangled_func_name

    return results


CUBIN_PATHS: dict[str, str] = {
    "sputnik_cuda_spmm": "intrasm_engine/3rdparty/sputnik/build/sputnik/spmm/CMakeFiles/cuda_spmm.dir/cuda_spmm.cu.cubin",
    "cutlass_tensorop_canonical": "cpp/small_projects/build_simt_tensorrt_canonical_cubin/build/cubin/tensorop_canonical.cubin",
    "cutlass_simt_canonical": "cpp/small_projects/build_simt_tensorrt_canonical_cubin/build/cubin/simt_canonical.cubin",
}
for key in CUBIN_PATHS:
    CUBIN_PATHS[key] = os.path.join(
        get_git_root_recursively(os.path.dirname(os.path.realpath(__file__))),
        CUBIN_PATHS[key],
    )


def refresh_sputnik_cuda_spmm_symbol_table():
    refresh_cuda_symbol_table(
        "sputnik_cuda_spmm_symbol_table.txt",
        CUBIN_PATHS["sputnik_cuda_spmm"],
    )


def get_sputnik_signature_to_mangle_name_map() -> dict[str, str]:
    return get_signature_to_mangle_name_map(
        "sputnik_cuda_spmm_symbol_table.txt"
    )


def refresh_cutlass_tensorop_canonical_symbol_table():
    refresh_cuda_symbol_table(
        "cutlass_tensorop_canonical_symbol_table.txt",
        CUBIN_PATHS["cutlass_tensorop_canonical"],
    )


def get_cutlass_tensorop_canonical_signature_to_mangle_name_map() -> (
    dict[str, str]
):
    return get_signature_to_mangle_name_map(
        "cutlass_tensorop_canonical_symbol_table.txt"
    )


def refresh_cutlass_simt_canonical_symbol_table():
    refresh_cuda_symbol_table(
        "cutlass_simt_canonical_symbol_table.txt",
        CUBIN_PATHS["cutlass_simt_canonical"],
    )


def get_cutlass_simt_canonical_signature_to_mangle_name_map() -> (
    dict[str, str]
):
    return get_signature_to_mangle_name_map(
        "cutlass_simt_canonical_symbol_table.txt"
    )


if __name__ == "__main__":
    refresh_sputnik_cuda_spmm_symbol_table()
    print(get_sputnik_signature_to_mangle_name_map())
    refresh_cutlass_simt_canonical_symbol_table()
    refresh_cutlass_tensorop_canonical_symbol_table()
