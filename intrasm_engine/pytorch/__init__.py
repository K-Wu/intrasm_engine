"""IntraSM Engine bindings for pyTorch"""

try:
    import torch

    torch._dynamo.config.error_on_nested_jit_trace = False
except:  # pylint: disable=bare-except
    pass
