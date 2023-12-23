if __name__ == "__main__":
    import pycuda.autoinit  # Initialize pycuda
    import pycuda.driver

    "Load the cubin file, which requires building sputnik with cubin enabled by configuring cmake with -DBUILD_CUBIN=ON"
    m = pycuda.driver.module_from_file(
        "/home/kwu/cupy-playground/intrasm_engine/3rdparty/sputnik/build/sputnik/spmm/CMakeFiles/cuda_spmm.dir/cuda_spmm.cu.cubin"
    )
    # void sputnik::<unnamed>::Kernel<sputnik::SpmmConfig<float, float, float4, (int)4, (int)8, (int)32, (int)8, (int)4, (int)0, (bool)0, (int)8>>(int, int, int, const int *, const T1::ScalarValue *, const int *, const T1::ScalarIndex *, const T1::ScalarValue *, const float *, T1::ScalarValue *)
    f = m.get_function(
        "_ZN7sputnik48_GLOBAL__N__a430860f_15_cuda_spmm_cu_cc_6504ba7b6KernelINS_10SpmmConfigIff6float4Li4ELi8ELi32ELi8ELi4ELi0ELb0ELi8EEEEEviiiPKiPKNT_11ScalarValueES6_PKNS7_11ScalarIndexESA_PKfPS8_"
    )
    import numpy as np
    import torch

    row_ptr = torch.zeros((3), device="cuda", dtype=torch.int32)
    f(
        np.int32(0),
        np.int32(0),
        np.int32(0),
        np.int64(0),
        np.int64(0),
        np.int64(0),
        np.int64(0),
        np.int64(0),
        np.int64(0),
        np.int64(0),
        block=(1, 1, 1),
        grid=(1, 1, 1),
    )
