from cutlass.op.gemm_grouped import (
    GroupedGemm,
    GemmGroupedArguments,
    GemmCoord,
)
from cutlass.op.gemm import (
    Gemm,
    GemmArguments,
    GemmCoord,
    DataType,
    GemmUniversalMode,
)
from cutlass.backend.evt import EpilogueFunctorVisitor
from cuda import cuda


def prepare_GemmArguments(
    plan: Gemm,
    A=None,
    B=None,
    C=None,
    D=None,
    alpha=None,
    beta=None,
    # sync: bool = True, # Unused argument in cutlass
    print_module: bool = False,
    visitor_args: dict = None,
    stream: cuda.CUstream = cuda.CUstream(0),
) -> GemmArguments:
    """This function is the first step Gemm.run() defined in cutlass/op/gemm.py cutlass python interface.
    We put them into this function to separate the execution from the argument preparation originally together in GropuedGemm.run().
    This allows us to use CUDAGraphConstructor. Otherwise, the compile in the preparation stage in run() forbid us to use CUDAGraphConstructor.
    The base cutlass Gemm.run() already supports void-C kernels. Just specify C as None. On the other hand, even though C is specified, it is not used in the kernel.
    """
    plan.run_setup()
    A = plan._verify_tensor(A, plan.A, plan._element_a, plan._layout_a, "A")
    B = plan._verify_tensor(B, plan.B, plan._element_b, plan._layout_b, "B")
    C = plan._verify_tensor(C, plan.C, plan._element_c, plan._layout_c, "C")
    D = plan._verify_tensor(D, plan.D, plan._element_d, plan._layout_d, "D")
    alpha = plan._verify_scalar(alpha, plan.alpha, plan._element_c, "alpha")
    beta = plan._verify_scalar(beta, plan.beta, plan._element_c, "beta")

    is_void_c = plan._element_c == DataType.void

    plan._verify_rank(A)
    plan._verify_rank(B)
    if not is_void_c:
        plan._verify_rank(C)
    plan._verify_rank(D)

    alignment_a = plan.possible_operations.find_alignment(
        A.shape, plan._layout_a, operand="A"
    )
    alignment_b = plan.possible_operations.find_alignment(
        B.shape, plan._layout_b, operand="B"
    )

    # Set C alignment based on D.shape so as to correctly get an alignment with void-C
    # kernels, for which `C` is None.
    alignment_c = plan.possible_operations.find_alignment(
        D.shape, plan._layout_c, operand="C"
    )
    plan.compile(
        plan._tile_description,
        alignment_A=alignment_a,
        alignment_B=alignment_b,
        alignment_C=alignment_c,
        print_module=print_module,
    )

    problem_size, mode, batch_count = plan._get_problem_args(A, B, C, D)

    if mode == GemmUniversalMode.Gemm or batch_count == 1:
        kwargs = {"split_k_slices": 1}
    else:
        kwargs = {
            "batch": batch_count,
            "batch_strides": {
                "A": plan._get_batch_stride(A),
                "B": plan._get_batch_stride(B),
                "C": plan._get_batch_stride(C),
                "D": plan._get_batch_stride(D),
            },
        }
    kwargs["stream"] = stream

    if isinstance(plan.epilogue_functor, EpilogueFunctorVisitor):
        output_op = plan.operation.epilogue_type(visitor_args)
    else:
        output_op = plan.operation.epilogue_type(alpha, beta)

    arguments = GemmArguments(
        operation=plan.operation,
        problem_size=problem_size,
        A=A,
        B=B,
        C=C,
        D=D,
        output_op=output_op,
        gemm_mode=mode,
        **kwargs,
    )
    return arguments


def prepare_GemmGroupedArguments(
    plan: GroupedGemm,
    A,
    B,
    C,
    D,
    alpha=None,
    beta=None,
    # sync: bool = True, # Unused argument in cutlass
    print_module: bool = False,
) -> GemmGroupedArguments:
    """This function is the first step GropuedGemm.run() defined in cutlass/op/gemm_grouped.py cutlass python interface.
    We put them into this function to separate the execution from the argument preparation originally together in GropuedGemm.run().
    This allows us to use CUDAGraphConstructor. Otherwise, the malloc in the preparation stage in run() forbid us to use CUDAGraphConstructor.
    Modified to support void-C kernels: just specify C as [None] * len(A).
    In current version, even though C is specified, it is not used in the kernel.
    """
    plan.run_setup()

    if len(A) != len(B) or len(A) != len(C) or len(A) != len(D):
        raise Exception("Lengths of A, B, C, and D lists must be equal")

    C_is_void = True
    for c in C:
        if c is not None:
            C_is_void = False
            break

    problem_sizes = []
    As, Bs, Cs, Ds = ([None] * len(A) for _ in range(4))
    for i in range(len(A)):
        As[i] = plan._verify_tensor(
            A[i], plan.A, plan._element_a, plan._layout_a, "A"
        )
        Bs[i] = plan._verify_tensor(
            B[i], plan.B, plan._element_b, plan._layout_b, "B"
        )
        if not C_is_void:
            Cs[i] = plan._verify_tensor(
                C[i], plan.C, plan._element_c, plan._layout_c, "C"
            )
        Ds[i] = plan._verify_tensor(
            D[i], plan.D, plan._element_d, plan._layout_d, "D"
        )
        problem_sizes.append(
            GemmCoord(A[i].shape[0], B[i].shape[1], A[i].shape[1])
        )

    alpha = plan._verify_scalar(alpha, plan.alpha, plan._element_c, "alpha")
    beta = plan._verify_scalar(beta, plan.beta, plan._element_c, "beta")

    alignment_a = min(
        (
            plan.possible_operations.find_alignment(
                A.shape, plan._layout_a, operand="A"
            )
            for A in As
        )
    )
    alignment_b = min(
        (
            plan.possible_operations.find_alignment(
                B.shape, plan._layout_b, operand="B"
            )
            for B in Bs
        )
    )

    plan_compile_kwargs = {
        "alignment_A": alignment_a,
        "alignment_B": alignment_b,
        "print_module": print_module,
    }
    if not C_is_void:
        alignment_c = min(
            (
                plan.possible_operations.find_alignment(
                    C.shape, plan._layout_c, operand="C"
                )
                for C in Cs
            )
        )
        plan_compile_kwargs["alignment_C"] = alignment_c
    plan.compile(
        plan.tile_description,
        **plan_compile_kwargs,
    )

    arguments = GemmGroupedArguments(
        operation=plan.operation,
        problem_sizes=problem_sizes,
        A=As,
        B=Bs,
        C=Cs,
        D=Ds,
        output_op=plan.operation.epilogue_type(alpha, beta),
    )

    return arguments
