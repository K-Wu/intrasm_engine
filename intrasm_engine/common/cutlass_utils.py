from cutlass.op.gemm_grouped import (
    GroupedGemm,
    GemmGroupedArguments,
    GemmCoord,
)


def prepare_GemmGroupedArguments(
    plan: GroupedGemm,
    A,
    B,
    C,
    D,
    alpha=None,
    beta=None,
    sync: bool = True,
    print_module: bool = False,
) -> GemmGroupedArguments:
    """This function is the first step GropuedGemm.run() defined in cutlass/op/gemm_grouped.py cutlass python interface.
    We put them into this function to separate the execution from the argument preparation originally together in GropuedGemm.run().
    This allows us to use CUDAGraphConstructor. Otherwise, the malloc in the preparation stage in run() forbid us to use CUDAGraphConstructor.
    """
    plan.run_setup()

    if len(A) != len(B) or len(A) != len(C) or len(A) != len(D):
        raise Exception("Lengths of A, B, C, and D lists must be equal")

    problem_sizes = []
    As, Bs, Cs, Ds = ([None] * len(A) for _ in range(4))
    for i in range(len(A)):
        As[i] = plan._verify_tensor(
            A[i], plan.A, plan._element_a, plan._layout_a, "A"
        )
        Bs[i] = plan._verify_tensor(
            B[i], plan.B, plan._element_b, plan._layout_b, "B"
        )
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
    alignment_c = min(
        (
            plan.possible_operations.find_alignment(
                C.shape, plan._layout_c, operand="C"
            )
            for C in Cs
        )
    )
    plan.compile(
        plan.tile_description,
        alignment_A=alignment_a,
        alignment_B=alignment_b,
        alignment_C=alignment_c,
        print_module=print_module,
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
