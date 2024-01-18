import torch
import intrasm_engine.common.triton_utils as triton_utils


def test_triton_matmul():
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    c = triton_utils.run_matmul(a, b)


def test_triton_matmul_fp16():
    a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    b = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    c = triton_utils.run_matmul(a, b)


if __name__ == "__main__":
    test_triton_matmul()
    test_triton_matmul_fp16()
