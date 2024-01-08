# From https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html
import os
import torch
from pathlib import Path

root_path: Path = Path(__file__).resolve().parent.parent
if __name__ == "__main__":
    assert os.path.normpath(os.getcwd()) == os.path.normpath(root_path)

    torch.classes.load_library(
        "intrasm_engine_extensions.cpython-311-x86_64-linux-gnu.so"
    )
    spmm_sputnik_atomic_op = torch.ops.iex_ops.spmm_sputnik_atomic
    spmm_sputnik_reuse_weight_op = torch.ops.iex_ops.spmm_sputnik_reuse_weight
    sddmm_sputnik_atomic_upd_weight_op = (
        torch.ops.iex_ops.sddmm_sputnik_atomic_upd_weight
    )

    def test_MyStackClass():
        stack_1 = (
            torch.classes.my_classes.MyStackClassExampleFactory().get_example()
        )
        stack_2 = torch.classes.my_classes.MyStackClass(["just", "testing"])
        stack_2.merge(stack_1)
        print("stack 1")
        for idx in range(2):
            print(stack_1.pop())
        print("stack 2")
        for idx in range(4):
            print(stack_2.pop())

    test_MyStackClass()
