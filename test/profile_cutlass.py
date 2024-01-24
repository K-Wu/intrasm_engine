from .test_interleaving import (
    gemm_tflops,
    _get_matmul_execs_cutlass_tensor_core_f32,
    _get_matmul_execs_cutlass_simt_f32,
    run_non_interleaving,
)
import cutlass
import argparse
import torch


def get_tile_description(use_tensorop=True):
    plan = cutlass.op.Gemm(
        element=torch.float32,
        layout_A=cutlass.LayoutType.ColumnMajor,
        layout_B=cutlass.LayoutType.RowMajor,
        layout_C=cutlass.LayoutType.ColumnMajor,
        element_C=cutlass.DataType.void,
        element_accumulator=cutlass.DataType.f32,
    )
    if not use_tensorop:
        plan.opclass = cutlass.OpcodeClass.Simt
    # for td in plan.tile_descriptions():
    #     print(td)
    return plan.tile_descriptions()


def get_single_line_str(td: cutlass.TileDescription):
    return f"{td.threadblock_shape},{td.warp_count},{td.stages}"


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-m", type=int, default=1024)
    argparser.add_argument("-n", type=int, default=1024)
    argparser.add_argument("-k", type=int, default=1024)
    args = argparser.parse_args()
    td_tensorop = get_tile_description(use_tensorop=True)
    for td in td_tensorop:
        try:
            matmul_exec = _get_matmul_execs_cutlass_tensor_core_f32(
                args.m, args.n, args.k, 1, td
            )
            time = run_non_interleaving(matmul_exec)
            print(
                "cutlass tensorop"
                f" {get_single_line_str(td)} {time} {gemm_tflops(args.m,args.n,args.k,time)}"
            )
        except Exception as e:
            print(
                "[skipped] Error when running cutlass tensorop"
                f" {get_single_line_str(td)} {e}"
            )

    td_simt = get_tile_description(use_tensorop=False)
    for td in td_simt:
        try:
            matmul_exec = _get_matmul_execs_cutlass_simt_f32(
                args.m, args.n, args.k, 1, td
            )
            time = run_non_interleaving(matmul_exec)
            print(
                "cutlass simt"
                f" {get_single_line_str(td)} {time} {gemm_tflops(args.m,args.n,args.k,time)}"
            )
        except Exception as e:
            print(
                "[skipped] Error when running cutlass simt"
                f" {get_single_line_str(td)} {e}"
            )
