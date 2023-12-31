from intrasm_engine.common.loading_utils import (
    refresh_cuda_symbol_table,
    get_signature_to_mangle_name_map,
)
from intrasm_engine.common.utils import (
    get_git_root_recursively,
)
import os

# These cubins are for test only. We suggest against the loading of these two cubins: Please use the official cutlass python interface insterad.
TEST_CUBIN_PATHS = {
    "cutlass_tensorop_canonical": "cpp/small_projects/build_simt_tensorrt_canonical_cubin/build/cubin/tensorop_canonical.cubin",
    "cutlass_simt_canonical": "cpp/small_projects/build_simt_tensorrt_canonical_cubin/build/cubin/simt_canonical.cubin",
}

for key in TEST_CUBIN_PATHS:
    TEST_CUBIN_PATHS[key] = os.path.join(
        get_git_root_recursively(os.path.dirname(os.path.realpath(__file__))),
        TEST_CUBIN_PATHS[key],
    )


def refresh_cutlass_tensorop_canonical_symbol_table():
    refresh_cuda_symbol_table(
        "cutlass_tensorop_canonical_symbol_table.txt",
        TEST_CUBIN_PATHS["cutlass_tensorop_canonical"],
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
        TEST_CUBIN_PATHS["cutlass_simt_canonical"],
    )


def get_cutlass_simt_canonical_signature_to_mangle_name_map() -> (
    dict[str, str]
):
    return get_signature_to_mangle_name_map(
        "cutlass_simt_canonical_symbol_table.txt"
    )


if __name__ == "__main__":
    # import sys
    import os
    import intrasm_engine.common.utils
    import intrasm_engine.common.loading_utils

    import os

    import pycuda.autoprimaryctx  # Initialize pycuda
    import pycuda.driver

    import numpy as np

    import torch

    "Load the cubin file, which requires building sputnik with cubin enabled by configuring cmake with -DBUILD_CUBIN=ON"
    m = pycuda.driver.module_from_file(
        os.path.join(
            intrasm_engine.common.utils.get_git_root_recursively(
                os.path.dirname(os.path.realpath(__file__)),
            ),
            "intrasm_engine/3rdparty/sputnik/build/sputnik/spmm/CMakeFiles/cuda_spmm.dir/cuda_spmm.cu.cubin",
        )
    )

    f = m.get_function(
        intrasm_engine.common.loading_utils.get_sputnik_signature_to_mangle_name_map()[
            "void sputnik::<unnamed>::Kernel<sputnik::SpmmConfig<float, float,"
            " float4, (int)4, (int)8, (int)32, (int)8, (int)4, (int)0,"
            " (bool)0, (int)8>>(int, int, int, const int *, const"
            " T1::ScalarValue *, const int *, const T1::ScalarIndex *, const"
            " T1::ScalarValue *, const float *, T1::ScalarValue *)"
        ]
    )

    row_ptr = torch.zeros((3), device="cuda", dtype=torch.int32)
    f(
        np.int32(0),
        np.int32(0),
        np.int32(0),
        row_ptr,
        np.int64(0),
        np.int64(0),
        np.int64(0),
        np.int64(0),
        np.int64(0),
        np.int64(0),
        block=(32, 1, 1),
        grid=(1, 1, 1),
    )

    refresh_cutlass_simt_canonical_symbol_table()
    refresh_cutlass_tensorop_canonical_symbol_table()
