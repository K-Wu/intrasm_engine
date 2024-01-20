from __future__ import annotations
from triton_autotuning.matmul_lib import (
    MatmulTiling,
    MatrixLayout,
    _matmul_kernel,
    _reduce_kernel,
    _get_common_type,
)
import torch
import triton
import triton.language as tl


def _run_matmul(
    M: int,
    N: int,
    K: int,
    s: torch.cuda.Stream,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    scratchpad: torch.Tensor,  # Largest size: c * SPLIT_K
    # Default values of the following arguments are provided in run_matmul()
    tiling: MatmulTiling,
    acc_ty: tl.dtype,  #
    ab_dtype: tl.dtype | None,  # None means no conversion during A dot B
    allow_tf32: bool,  # This flag enables or disables tensor core for multiply-accumulate
):
    """Based on benchmark_matmul_tiling in triton_autotuning/matmul_lib.py"""

    with torch.cuda.stream(s):
        grid = lambda META: (  # pylint: disable=g-long-lambda
            triton.cdiv(M, tiling.BLOCK_M) * triton.cdiv(N, tiling.BLOCK_N),
            tiling.SPLIT_K,
            1,  # batch
        )

        m, n, k = int(M), int(N), int(K)
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
            num_warps=tiling.num_warps,  # Triton JIT kernel grid config
            num_stages=tiling.num_stages,  # Triton JIT kernel software-pipeline config
            force_num_warps=tiling.num_warps,  # unused kernel arguments to force recompilation on different configruations
            force_num_stages=tiling.num_stages,  # unused kernel arguments to force recompilation on different configruations
            acc_ty=acc_ty,
            IS_ATOMIC_ADD=False,
            AB_DTYPE=ab_dtype,
            ALLOW_TF32=allow_tf32,
        )
        if tiling.SPLIT_K != 1:
            # Run reduction kernel.
            _reduce_kernel[(triton.cdiv(M * N, 1024),)](
                scratchpad,
                c,
                # The original code is int(dims.M). We filed an issue https://github.com/tensorflow/tensorflow/issues/62789
                row_size=int(M) * int(N),
                col_size=tiling.SPLIT_K,
                num_stages=1,  # Triton JIT kernel software-pipeline config
                num_warps=1024 // 32,  # Triton JIT kernel grid config
                row_block_size=1024,
            )


def run_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    c_ty: torch.dtype | None = None,
    acc_ty: tl.dtype | None = None,
    ab_dtype: tl.dtype | None = None,  # None means no conversion during A\dotB
    allow_tf32: bool = False,
    tiling: MatmulTiling = MatmulTiling(
        128,
        128,
        16,
        1,
        MatrixLayout.ROW_MAJOR,
        MatrixLayout.ROW_MAJOR,
        MatrixLayout.ROW_MAJOR,
        3,
        4,
    ),
):
    if c_ty is None:
        c_ty = _get_common_type(a.dtype, b.dtype)
    if acc_ty is None:
        # acc_ty = getattr(tl, str(c_ty).split(".")[1]) # tl.dot by default produces fp32 even though a and b are fp16
        acc_ty = tl.float32
    c = torch.zeros((a.shape[0], b.shape[1]), dtype=c_ty, device=a.device)
    if tiling.SPLIT_K > 1:
        # Create scratchpad.
        scratchpad = torch.zeros(
            (tiling.SPLIT_K, a.shape[0] * b.shape[1]),
            dtype=c_ty,
            device=a.device,
        )
    else:
        scratchpad = torch.zeros(
            (1, 1), dtype=c_ty, device=a.device
        )  # Suppress type checker

    current_stream = torch.cuda.current_stream()
    _run_matmul(
        a.shape[0],
        b.shape[1],
        a.shape[1],
        current_stream,
        a,
        b,
        c,
        scratchpad,
        ab_dtype=ab_dtype,
        allow_tf32=allow_tf32,
        tiling=tiling,
        acc_ty=acc_ty,
    )
    return c
