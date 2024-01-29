# From https://pytorch.org/tutorials/prototype/semi_structured_sparse.html

import torch
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
from torch.utils.benchmark import Timer

SparseSemiStructuredTensor._FORCE_CUTLASS = True

# mask Linear weight to be 2:4 sparse
mask = torch.Tensor([0, 0, 1, 1]).tile((3072, 2560)).cuda().bool()
linear = torch.nn.Linear(10240, 3072).half().cuda().eval()
# linear = torch.nn.Linear(10240, 3072).cuda().eval()
linear.weight = torch.nn.Parameter(mask * linear.weight)

x = torch.rand(3072, 10240).half().cuda()
# float32 not supported yet: x = torch.rand(3072, 10240).cuda()

with torch.inference_mode():
    dense_output = linear(x)
    dense_t = (
        Timer(stmt="linear(x)", globals={"linear": linear, "x": x})
        .blocked_autorange()
        .median
        * 1e3
    )

    # accelerate via SparseSemiStructuredTensor
    linear.weight = torch.nn.Parameter(
        # Error original_tensor.dtype torch.float32 is not a supported dtype! dtype must be one of: {_DTYPE_TO_SEMI_STRUCTURED_SPARSE_CONFIG}
        to_sparse_semi_structured(linear.weight)
    )

    sparse_output = linear(x)
    sparse_t = (
        Timer(stmt="linear(x)", globals={"linear": linear, "x": x})
        .blocked_autorange()
        .median
        * 1e3
    )

    # sparse and dense matmul are numerically equivalent
    assert torch.allclose(sparse_output, dense_output, atol=1e-3)
    print(
        f"Dense: {dense_t:.3f}ms Sparse: {sparse_t:.3f}ms | Speedup:"
        f" {(dense_t / sparse_t):.3f}x"
    )
