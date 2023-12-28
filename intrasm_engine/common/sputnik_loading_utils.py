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


def refresh_sputnik_cuda_spmm_symbol_table():
    symbol_table_filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "sputnik_cuda_spmm_symbol_table.txt",
    )
    cubin_filename = os.path.join(
        get_git_root_recursively(symbol_table_filename),
        "intrasm_engine/3rdparty/sputnik/build/sputnik/spmm/CMakeFiles/cuda_spmm.dir/cuda_spmm.cu.cubin",
    )
    # Use cuobjdump -symbols to get the symbol table
    subprocess.check_call(
        [
            "cuobjdump",
            "-symbols",
            cubin_filename,
        ],
        stdout=open(symbol_table_filename, "w"),
    )
    return


def get_signature_to_mangle_name_map() -> dict[str, str]:
    """This function returns a dictionary with key being the signature of the function, and value being the mangled name of the function.
    This is non-trivial because mangle name contains certain random strings, e.g., when dealing with unnamed namespace.
    """
    results: dict[str, str] = {}
    # Read the symbol table .txt at the same directory as this script
    with open(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "sputnik_cuda_spmm_symbol_table.txt",
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


if __name__ == "__main__":
    refresh_sputnik_cuda_spmm_symbol_table()
    print(get_signature_to_mangle_name_map())
