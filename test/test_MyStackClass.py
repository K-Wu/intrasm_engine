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
    torch.classes.my_classes.MyStackClass(["just", "testing"])
