from triton_autotuning.matmul_lib import (
    MatmulSize,
    MatmulTiling,
    _matmul_kernel,
    _reduce_kernel,
)
import torch
import triton
import triton.language as tl


def run_matmul(
    dims: MatmulSize,
    tiling: MatmulTiling,
    s: torch.cuda.Stream,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    scratchpad: torch.Tensor,  # Largest size: c * SPLIT_K
    acc_ty: tl.dtype = tl.float32,
):
    """Based on benchmark_matmul_tiling in triton_autotuning/matmul_lib.py"""

    with torch.cuda.stream(s):
        grid = lambda META: (  # pylint: disable=g-long-lambda
            triton.cdiv(dims.M, tiling.BLOCK_M)
            * triton.cdiv(dims.N, tiling.BLOCK_N),
            tiling.SPLIT_K,
            1,  # batch
        )

        m, n, k = int(dims.M), int(dims.N), int(dims.K)
        used_output = c if tiling.SPLIT_K == 1 else scratchpad
        _matmul_kernel[grid](
            a,
            b,
            used_output,
            m=m,
            n=n,
            k=k,
            # We assume a, b, c are all row-major for now as our top priority is Pytorch.
            stride_am=k,
            stride_ak=1,
            stride_bk=n,
            stride_bn=1,
            stride_cm=n,
            stride_cn=1,
            block_m=int(tiling.BLOCK_M),
            block_n=int(tiling.BLOCK_N),
            block_k=int(tiling.BLOCK_K),
            group_m=8,
            split_k=tiling.SPLIT_K,
            force_num_warps=tiling.num_warps,  # unused
            force_num_stages=tiling.num_stages,  # unused
            acc_ty=acc_ty,
            IS_ATOMIC_ADD=False,
        )
        if tiling.SPLIT_K != 1:
            # Run reduction kernel.
            _reduce_kernel[(triton.cdiv(dims.M * dims.N, 1024),)](
                scratchpad,
                c,
                # The original code is int(dims.M). We filed an issue https://github.com/tensorflow/tensorflow/issues/62789
                row_size=int(dims.M) * int(dims.N),
                col_size=tiling.SPLIT_K,
                row_block_size=1024,
            )
