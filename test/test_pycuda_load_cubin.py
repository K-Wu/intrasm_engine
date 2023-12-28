if __name__ == "__main__":
    # import sys
    import os
    import intrasm_engine.common.utils
    import intrasm_engine.common.sputnik_loading_utils

    # sys.path.append(
    #     os.path.join(
    #         intrasm_engine.common.utils.get_git_root_recursively(
    #             os.path.dirname(os.path.realpath(__file__)),
    #         ),
    #         "intrasm_engine/3rdparty/SparTA",
    #     )
    # )
    # import sparta  # Initialize pycuda and torch
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
        intrasm_engine.common.sputnik_loading_utils.get_signature_to_mangle_name_map()[
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
