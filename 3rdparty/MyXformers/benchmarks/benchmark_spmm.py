# Based on https://github.com/facebookresearch/xformers/blob/dd96b8d8beda5308fb433c1ef3ff04b7f178c263/xformers/benchmarks/benchmark_sddmm.py
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import torch
from torch.utils import benchmark

from xformers.components.attention._sputnik_sparse import _csr_to_coo
from xformers.components.attention.core import (
    SparseCS,
    _create_random_sparsity,
)

MIN_RUN_TIME = 0.2


def _get_fn(backend):
    if backend == "csr_ge":
        raise NotImplementedError("xformer has no csr_ge backend for spmm")
    elif backend == "csr_sputnik":
        fn = torch.ops.xformers.spmm_sputnik
    elif backend == "coo_ge":
        raise NotImplementedError("xformer has no coo_ge backend for spmm")
    elif backend == "csr_to_coo":
        raise NotImplementedError(
            "csr_to_coo's arguments are different from spmm_sputnik's. Please"
            " get csr_to_coo performance in bench_sddmm.py instead."
        )

    return fn


def bench_spmm(configs):
    min_run_time = MIN_RUN_TIME

    device = torch.device("cuda")
    results = []

    for (B, M, K), prob in configs:
        b = torch.rand(B, K, M, device=device)

        a_sparse = _create_random_sparsity(
            torch.rand(1, M, K), prob, divisible_by=16
        )
        a_sparse = a_sparse.repeat(B, 1, 1)
        bb = b
        a_sparse = SparseCS(a_sparse, device)
        row_indices = a_sparse.row_indices
        row_offsets = a_sparse.row_offsets
        column_indices = a_sparse.column_indices

        for backend in ["csr_sputnik"]:
            fn_str = (
                "fn(b, row_indices, values, row_offsets, column_indices, m)"
            )
            fn = _get_fn(backend)
            results.append(
                benchmark.Timer(
                    stmt=fn_str,
                    globals={
                        "b": bb,
                        "row_indices": row_indices,
                        "values": a_sparse.values,
                        "row_offsets": row_offsets,
                        "column_indices": column_indices,
                        "fn": fn,
                        "m": M,
                    },
                    label="spmm",
                    sub_label=(
                        f"B={B:>4d}, M=N={M:>4d}, K={K:>3d}, prob={prob:0.4f}"
                    ),
                    description=backend,
                ).blocked_autorange(min_run_time=min_run_time)
            )

    compare = benchmark.Compare(results)
    compare.print()
    return results


# batch size 32, for different layers
SWIN_T_SIZES = [(96, 3136, 32), (192, 784, 32), (384, 196, 32), (768, 49, 32)]
swin_t_config = list(zip(SWIN_T_SIZES, (0.9844, 0.9375, 0.75, 0.0)))

# some random values
BASIC_SIZES = [(32, 1024, 32), (32, 1024, 128), (8, 4096, 32), (8, 4096, 128)]
SPARSITIES = [0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]
basic_config = list(itertools.product(BASIC_SIZES, SPARSITIES))

# batch size 32 here
vit_sizes = [
    (192, 785, 64),  # deit_small_patch8_224
    (192, 197, 64),  # deit_small_patch16_224
    (384, 785, 64),  # deit_base_patch8_224
    (384, 197, 64),  # deit_base_patch16_224
]
SPARSITIES = [0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97]
vit_config = list(itertools.product(vit_sizes, SPARSITIES))

results = []

print("Swin Transformer")
results += bench_spmm(swin_t_config)
print("ViT")
results += bench_spmm(vit_config)
print("Basic cases")
results += bench_spmm(basic_config)
