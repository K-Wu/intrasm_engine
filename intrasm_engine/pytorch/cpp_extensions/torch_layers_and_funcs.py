import torch


# Implementation of torch matmul autograd functions
class MyLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, **kwargs):
        constructor = kwargs["constructor"]
        backward_constructor = kwargs["backward_constructor"]
        num_streams = kwargs["num_streams"]
        assert num_streams == 1
        ctx.save_for_backward(input, weight)
        ctx.constructor = constructor
        ctx.backward_constructor = backward_constructor
        ctx.num_streams = num_streams
        # Conforming to torch.nn.Linear, weight is (out_features, in_features)
        output = input.mm(weight.t())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        num_streams = ctx.num_streams
        backward_constructor = ctx.backward_constructor
        grad_input = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        return grad_input, grad_weight


class MyLinearPartitioned(torch.autograd.Function):
    ...
    # TODO: implement this
