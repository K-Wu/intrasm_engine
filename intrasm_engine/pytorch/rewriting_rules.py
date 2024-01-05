import torch


def make_single_kernel_autograd_function(
    func_class: type[torch.autograd.Function],
) -> None:
    """We need to register the static methods to the autograd functions to resolve the reference to the classes: if we use __class__ in the parent class, all child class will refer to the parent class, which is not what we want."""

    # apply() takes no keyword arguments. Adding **kwargs to make linter happy.
    @staticmethod
    def forward(ctx, *args, **kwargs):
        outputs, tensors_to_save = func_class._forward(ctx, *args, **kwargs)
        ctx.save_for_backward(*tensors_to_save)
        return (*outputs,)

    @staticmethod
    def backward(ctx, *args, **kwargs):
        tensors_to_save = ctx.saved_tensors
        return func_class._backward(ctx, *args, *tensors_to_save, **kwargs)

    """
    @staticmethod
    def _forward(ctx, *args, **kwargs) -> tuple[tuple, tuple]:
        This method returns a tuple of outputs and tensors_to_save for backward.
        We do not save_for_backward on the tensors because save_for_backward can only be executed once.
        In the KernelPairAutogradFunction, we have to execute _forward twice. Our solution is to collect the tensors to save in the two _forward() function calls, and save the tensors only once in the forward() function.
    """
    assert "_forward" in func_class.__dict__

    """
    @staticmethod
    def _backward(ctx, *args, **kwargs):
        This method actually takes in ctx, *args, *tensors_to_save, **kwargs.
        In the KernelPairAutogradFunction, the saved tensor is a combination of the tensors_to_save from the two _forward() function calls. We have to unpack them in the backward() function, and pass them to _backward() function.
    """
    assert "_backward" in func_class.__dict__

    assert "num_inputs" in func_class.__dict__
    assert "num_saved_tensors" in func_class.__dict__

    func_class.forward = forward
    func_class.backward = backward


def make_kernel_pair_autograd_function(
    func1: type[torch.autograd.Function], func2: type[torch.autograd.Function]
) -> type[torch.autograd.Function]:
    class KernelPairAutogradFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args, **kwargs):
            num_args_1 = func1.num_inputs
            outputs_1, tensors_to_save_1 = func1._forward(
                ctx, *args[:num_args_1], **kwargs
            )
            outputs_2, tensors_to_save_2 = func2._forward(
                ctx, *args[num_args_1:], **kwargs
            )
            ctx.save_for_backward(*tensors_to_save_1, *tensors_to_save_2)
            return (*outputs_1, *outputs_2)

        @staticmethod
        def backward(ctx, *args, **kwargs):
            tensors_to_save = ctx.saved_tensors
            tensors_to_save_1 = tensors_to_save[: func1.num_saved_tensors]
            tensors_to_save_2 = tensors_to_save[func1.num_saved_tensors :]
            gradients_1 = func1._backward(
                ctx, *args[: func1.num_inputs], *tensors_to_save_1, **kwargs
            )
            gradients_2 = func2._backward(
                ctx, *args[func1.num_inputs :], *tensors_to_save_2, **kwargs
            )
            return (*gradients_1, *gradients_2)

    return KernelPairAutogradFunction
