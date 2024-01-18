import torch
from .layers_and_funcs_utils import MyAutogradFunc


# Implementation of torch matmul autograd functions
class MyLinear(MyAutogradFunc):
    @staticmethod
    def forward(ctx, input, weight, **kwargs):
        ctx.save_for_backward(input, weight)
        ctx.constructor = kwargs["constructor"]
        ctx.backward_constructor = kwargs["backward_constructor"]
        ctx.num_streams = kwargs["num_streams"]
        ctx.constructor_enabled = kwargs["constructor_enabled"]
        ctx.stream_beg = kwargs["stream_beg"]
        assert ctx.num_streams == 1
        # Conforming to torch.nn.Linear, weight is (out_features, in_features)
        output = input.mm(weight.t())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        num_streams = ctx.num_streams
        backward_constructor = ctx.backward_constructor
        constructor_enabled = ctx.constructor_enabled
        grad_input = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        return grad_input, grad_weight


class MyLinearPartitioned(MyAutogradFunc):
    ...
    # TODO: implement this
