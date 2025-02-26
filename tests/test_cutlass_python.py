"""This is the code from https://github.com/NVIDIA/cutlass/blob/8236f30675bbe98f81d11c05764b77bfcb25b8cc/examples/python/02_pytorch_extension_grouped_gemm.ipynb.
We use the code here to demonstrate how to use cutlass python interface, and test if the cutlass python interface is working.
"""

import cutlass
import torch


def test_cutlass_gemm():
    dtype = torch.float16
    plan = cutlass.op.Gemm(
        element=dtype,
        layout=cutlass.LayoutType.RowMajor,
        element_C=cutlass.DataType.void,
        element_accumulator=cutlass.DataType.f16,
    )
    import random

    random.seed(2023)

    # Utility function to initialize A, B, C, and D matrices corresponding to dimensions M, N, and K
    def initialize(dtype, M, N, K):
        sizes = [(M, K), (K, N), (M, N)]
        return [
            torch.randint(-3, 3, size, device="cuda").to(dtype)
            for size in sizes
        ]

    # Utility function to generate `problems` GEMMs of random sizes
    def generate_problems():
        valid_sizes = [128, 256, 512, 1024]
        M, N, K = [random.choice(valid_sizes) for _ in range(3)]
        A, B, D = initialize(dtype, M, N, K)
        return A, B, D

    (
        A,
        B,
        D,
    ) = generate_problems()

    plan.run(A, B, None, D, print_module=False)
    D_torch = A @ B

    assert torch.allclose(D, D_torch)


def print_cutlass_f16_simt_td():
    dtype = torch.float16
    plan = cutlass.op.Gemm(
        element=dtype,
        layout=cutlass.LayoutType.RowMajor,
        element_C=cutlass.DataType.void,
        element_accumulator=cutlass.DataType.f16,
    )
    plan.opclass = cutlass.OpcodeClass.Simt
    for td in plan.tile_descriptions():
        print(td)


def test_customize_cutlass_gemm_td():
    dtype = torch.float32
    plan = cutlass.op.Gemm(
        element=dtype,
        layout=cutlass.LayoutType.RowMajor,
        element_C=cutlass.DataType.void,
        element_accumulator=cutlass.DataType.f32,
    )
    plan.opclass = cutlass.OpcodeClass.Simt
    for td in plan.tile_descriptions():
        print(td)
    plan.tile_description = {
        "threadblock_shape": [128, 64, 8],
        "warp_count": [2, 2, 1],
        "stages": 5,
    }

    import random

    random.seed(2023)

    # Utility function to initialize A, B, C, and D matrices corresponding to dimensions M, N, and K
    def initialize(dtype, M, N, K):
        sizes = [(M, K), (K, N), (M, N)]
        return [
            torch.randint(-3, 3, size, device="cuda").to(dtype)
            for size in sizes
        ]

    # Utility function to generate `problems` GEMMs of random sizes
    def generate_problems():
        valid_sizes = [128, 256, 512, 1024]
        M, N, K = [random.choice(valid_sizes) for _ in range(3)]
        A, B, D = initialize(dtype, M, N, K)
        return A, B, D

    (
        A,
        B,
        D,
    ) = generate_problems()

    plan.run(A, B, None, D, print_module=False)
    D_torch = A @ B

    assert torch.allclose(D, D_torch)


def test_cutlass_grouped_gemm():
    dtype = torch.float16
    plan = cutlass.op.GroupedGemm(
        element=dtype,
        layout=cutlass.LayoutType.RowMajor,
    )
    import random

    random.seed(2023)

    # Utility function to initialize A, B, C, and D matrices corresponding to dimensions M, N, and K
    def initialize(dtype, M, N, K):
        sizes = [(M, K), (K, N), (M, N), (M, N)]
        return [
            torch.randint(-3, 3, size, device="cuda").to(dtype)
            for size in sizes
        ]

    # Utility function to generate `problems` GEMMs of random sizes
    def generate_problems(problems):
        valid_sizes = [128, 256, 512, 1024]
        As, Bs, Cs, Ds = [], [], [], []
        for _ in range(problems):
            M, N, K = [random.choice(valid_sizes) for _ in range(3)]
            A, B, C, D = initialize(dtype, M, N, K)
            As.append(A)
            Bs.append(B)
            Cs.append(C)
            Ds.append(D)
        return As, Bs, Cs, Ds

    (
        As,
        Bs,
        Cs,
        Ds,
    ) = generate_problems(50)

    plan.run(As, Bs, Cs, Ds, print_module=False)
    Ds_torch = [a @ b for a, b in zip(As, Bs)]

    for d, d_torch in zip(Ds, Ds_torch):
        assert torch.allclose(d, d_torch)

    def test_emit_and_compile():
        op = plan.construct()
        grouped_gemm = cutlass.emit.pytorch(
            op, name="grouped_gemm", cc=plan.cc, sourcedir="out", jit=True
        )

    # Emitting and compilation are not working yet due to environment variable issue in Cutlass.
    # test_emit_and_compile()


if __name__ == "__main__":
    print_cutlass_f16_simt_td()
    print("Done print_cutlass_f16_simt_td")
    test_customize_cutlass_gemm_td()
    test_cutlass_gemm()
    test_cutlass_grouped_gemm()
