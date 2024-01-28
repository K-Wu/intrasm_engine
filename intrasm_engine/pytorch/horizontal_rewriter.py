from __future__ import annotations
import torch
from torch import nn
from .cpp_extensions.layers_and_funcs.utils import MyAutogradFunc
from .cpp_extensions.cuda_graph_constructor import TorchCUDAGraphConstructor
from sparta.common.tuning import TunableItemCfg
from typing import Any, Optional


def simple_combine_search_space(
    space_1: dict[str, TunableItemCfg],
    space_2: dict[str, TunableItemCfg],
    space_3: dict[str, TunableItemCfg],
) -> dict[str, TunableItemCfg]:
    results: dict[str, TunableItemCfg] = {}
    for key in space_1:
        results["space_1." + key] = space_1[key]
    for key in space_2:
        results["space_2." + key] = space_2[key]
    for key in space_3:
        results["space_3." + key] = space_3[key]
    return results


def simple_unpack_params(
    params: dict[str, Any]
) -> (
    tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
    | tuple[dict[str, Any], dict[str, Any]]
):
    params_1 = {}
    params_2 = {}
    params_3 = {}
    for key in params:
        if key.startswith("space_1."):
            params_1[key[8:]] = params[key]
        elif key.startswith("space_2."):
            params_2[key[8:]] = params[key]
        elif key.startswith("space_3."):
            params_3[key[8:]] = params[key]
        else:
            raise ValueError(f"Unknown key {key}")
    if len(params_3) == 0:
        return params_1, params_2
    else:
        return params_1, params_2, params_3


def make_single_kernel_autograd_function(
    func_class: type[MyAutogradFunc],
) -> None:
    """We need to register the static methods to the autograd functions to resolve the reference to the classes: if we use __class__ in the parent class, all child class will refer to the parent class, which is not what we want."""

    # apply() takes no keyword arguments. Adding **kwargs to make linter happy.
    @staticmethod
    def forward(ctx, *args, **kwargs):
        fwd_kwargs_unwrapped_1 = kwargs["fwd_kwargs_1"]
        outputs, ctx_dict_to_save, tensors_to_save = func_class._forward(
            *args, **fwd_kwargs_unwrapped_1
        )
        ctx.save_for_backward(*tensors_to_save)
        ctx["ctx_dict_1"] = ctx_dict_to_save
        return (*outputs,)

    @staticmethod
    def backward(ctx, *grad_args):
        tensors_to_save = ctx.saved_tensors
        return func_class._backward(
            *grad_args, *tensors_to_save, **ctx["ctx_dict_1"]
        )

    func_class.forward = forward
    func_class.backward = backward


def make_kernel_pair_or_triple_autograd_function(
    func1: type[MyAutogradFunc],
    func2: type[MyAutogradFunc],
    func3: type[MyAutogradFunc] | None = None,
) -> type[MyAutogradFunc]:
    class KernelPairOrTripleAutogradFunction(MyAutogradFunc):
        @staticmethod
        def num_inputs(**fwd_kwargs_unwrapped) -> int:
            return (
                func1.num_inputs(**fwd_kwargs_unwrapped)
                + func2.num_inputs(**fwd_kwargs_unwrapped)
                + (func3.num_inputs(**fwd_kwargs_unwrapped) if func3 else 0)
            )

        @staticmethod
        def num_outputs(**ctx_kwargs_unwrapped) -> int:
            return (
                func1.num_outputs(**ctx_kwargs_unwrapped)
                + func2.num_outputs(**ctx_kwargs_unwrapped)
                + (func3.num_outputs(**ctx_kwargs_unwrapped) if func3 else 0)
            )

        @staticmethod
        def num_saved_tensors(**ctx_kwargs_unwrapped) -> int:
            return (
                func1.num_saved_tensors(**ctx_kwargs_unwrapped)
                + func2.num_saved_tensors(**ctx_kwargs_unwrapped)
                + (
                    func3.num_saved_tensors(**ctx_kwargs_unwrapped)
                    if func3
                    else 0
                )
            )

        @classmethod
        def get_search_space(
            cls,
            sample_inputs: list[torch.Tensor],
            sample_grads: Optional[list[torch.Tensor]],
        ) -> dict[str, TunableItemCfg]:  # {'param_name': {param_values}}
            sample_input_1 = sample_inputs[: func1.num_inputs()]
            sample_input_2 = sample_inputs[
                func1.num_inputs() : func1.num_inputs() + func2.num_inputs()
            ]
            sample_grads_1 = None
            sample_grads_2 = None
            if sample_grads is not None:
                sample_grads_1 = sample_grads[: func1.num_outputs()]
                sample_grads_2 = sample_grads[
                    func1.num_outputs() : func1.num_outputs()
                    + func2.num_outputs()
                ]
            search_space_1 = func1.get_search_space(
                sample_input_1, sample_grads_1
            )
            search_space_2 = func2.get_search_space(
                sample_input_2, sample_grads_2
            )
            search_space_3 = {}
            if func3 is not None:
                sample_input_3 = sample_inputs[
                    func1.num_inputs() + func2.num_inputs() :
                ]
                sample_grads_3 = None
                if sample_grads is not None:
                    sample_grads_3 = sample_grads[
                        func1.num_outputs() + func2.num_outputs() :
                    ]
                search_space_3 = func3.get_search_space(
                    sample_input_3, sample_grads_3
                )
            return simple_combine_search_space(
                search_space_1, search_space_2, search_space_3
            )

        @classmethod
        def check_fwd_kwargs_sanity(cls, fwd_kwargs: dict[str, Any]):
            assert "fwd_kwargs_1" in fwd_kwargs
            assert "fwd_kwargs_2" in fwd_kwargs
            if func3 is not None:
                assert "fwd_kwargs_3" in fwd_kwargs
                func3._check_unwrapped_fwd_kwargs_sanity(
                    fwd_kwargs["fwd_kwargs_3"]
                )
            func1._check_unwrapped_fwd_kwargs_sanity(
                fwd_kwargs["fwd_kwargs_1"]
            )
            func2._check_unwrapped_fwd_kwargs_sanity(
                fwd_kwargs["fwd_kwargs_2"]
            )

        @classmethod
        def get_test_fwd_kwargs(cls, params: dict) -> dict[str, Any]:
            results = {}
            fwd_kwargs_3 = None
            if func3 is not None:
                params_tuples = simple_unpack_params(params)
                assert len(params_tuples) == 3
                params_1, params_2, params_3 = params_tuples
                fwd_kwargs_3 = func3.get_test_fwd_kwargs(params_3)
            else:
                params_tuples = simple_unpack_params(params)
                assert len(params_tuples) == 2
                params_1, params_2 = params_tuples
            fwd_kwargs_1 = func1.get_test_fwd_kwargs(params_1)
            fwd_kwargs_2 = func2.get_test_fwd_kwargs(params_2)
            results["fwd_kwargs_1"] = fwd_kwargs_1["fwd_kwargs_1"]
            results["fwd_kwargs_2"] = fwd_kwargs_2["fwd_kwargs_1"]
            if func3 is not None:
                assert fwd_kwargs_3 is not None
                results["fwd_kwargs_3"] = fwd_kwargs_3["fwd_kwargs_1"]
            return results

        @classmethod
        def check_params_sanity(cls, params: dict):
            if func3 is not None:
                params_tuples = simple_unpack_params(params)
                assert len(params_tuples) == 3
                params_1, params_2, params_3 = params_tuples
                func3.check_params_sanity(params_3)
            else:
                params_tuples = simple_unpack_params(params)
                assert len(params_tuples) == 2
                params_1, params_2 = params_tuples
            func1.check_params_sanity(params_1)
            func2.check_params_sanity(params_2)

        @classmethod
        def build_and_test(
            cls,
            sample_inputs: list[torch.Tensor],
            sample_grads: Optional[list[torch.Tensor]],
            params: dict,
        ):
            sample_inputs_1 = sample_inputs[: func1.num_inputs()]
            sample_inputs_2 = sample_inputs[
                func1.num_inputs() : func1.num_inputs() + func2.num_inputs()
            ]
            sample_grads_1 = None
            sample_grads_2 = None
            if sample_grads is not None:
                sample_grads_1 = sample_grads[: func1.num_outputs()]
                sample_grads_2 = sample_grads[
                    func1.num_outputs() : func1.num_outputs()
                    + func2.num_outputs()
                ]
            if func3 is not None:
                sample_inputs_3 = sample_inputs[
                    func1.num_inputs() + func2.num_inputs() :
                ]
                sample_grads_3 = None
                if sample_grads is not None:
                    sample_grads_3 = sample_grads[
                        func1.num_outputs() + func2.num_outputs() :
                    ]
            # TODO

        @staticmethod
        def forward(ctx, *args, **kwargs):
            fwd_kwargs_unwrapped_1 = kwargs["fwd_kwargs_1"]
            fwd_kwargs_unwrapped_2 = kwargs["fwd_kwargs_2"]
            num_args_1 = func1.num_inputs(**fwd_kwargs_unwrapped_1)
            outputs_1, ctx_dict_to_save_1, tensors_to_save_1 = func1._forward(
                *args[:num_args_1], **fwd_kwargs_unwrapped_1
            )
            outputs_2, ctx_dict_to_save_2, tensors_to_save_2 = func2._forward(
                *args[num_args_1:], **fwd_kwargs_unwrapped_2
            )
            if func3 is not None:
                fwd_kwargs_unwrapped_3 = kwargs["fwd_kwargs_3"]
                num_args_2 = func2.num_inputs(**fwd_kwargs_unwrapped_2)
                (
                    outputs_3,
                    ctx_dict_to_save_3,
                    tensors_to_save_3,
                ) = func3._forward(
                    *args[num_args_1 + num_args_2 :], **fwd_kwargs_unwrapped_3
                )
                ctx.save_for_backward(
                    *tensors_to_save_1,
                    *tensors_to_save_2,
                    *tensors_to_save_3,
                )
                ctx["ctx_dict_1"] = ctx_dict_to_save_1
                ctx["ctx_dict_2"] = ctx_dict_to_save_2
                ctx["ctx_dict_3"] = ctx_dict_to_save_3
                return (*outputs_1, *outputs_2, *outputs_3)
            else:
                ctx.save_for_backward(*tensors_to_save_1, *tensors_to_save_2)
                ctx["ctx_dict_1"] = ctx_dict_to_save_1
                ctx["ctx_dict_2"] = ctx_dict_to_save_2
                return (*outputs_1, *outputs_2)

        @staticmethod
        def backward(ctx, *grad_args):
            tensors_to_save = ctx.saved_tensors
            tensors_to_save_1 = tensors_to_save[
                : func1.num_saved_tensors(**ctx["ctx_dict_1"])
            ]
            gradients_1 = func1._backward(
                *grad_args[: func1.num_outputs(**ctx["ctx_dict_1"])],
                *tensors_to_save_1,
                **ctx["ctx_dict_1"],
            )
            tensors_to_save_2 = tensors_to_save[
                func1.num_saved_tensors(**ctx["ctx_dict_1"]) :
            ]
            gradients_2 = func2._backward(
                ctx,
                *grad_args[
                    func1.num_outputs(**ctx["ctx_dict_1"]) : func1.num_outputs(
                        **ctx["ctx_dict_1"]
                    )
                    + func2.num_outputs(**ctx["ctx_dict_2"])
                ],
                *tensors_to_save_2,
                **ctx["ctx_dict_2"],
            )
            if func3 is not None:
                tensors_to_save_3 = tensors_to_save_2[
                    func2.num_saved_tensors(**ctx["ctx_dict_2"]) :
                ]
                gradients_3 = func3._backward(
                    ctx,
                    *grad_args[
                        func1.num_outputs(**ctx["ctx_dict_1"])
                        + func2.num_outputs(**ctx["ctx_dict_2"]) :
                    ],
                    *tensors_to_save_3,
                    **ctx["ctx_dict_3"],
                )
                return (*gradients_1, *gradients_2, *gradients_3)
            else:
                return (*gradients_1, *gradients_2)

    return KernelPairOrTripleAutogradFunction


class CUDAGraphModulePreviousLayerFunction(torch.autograd.Function):
    """The layer that replays the graph captured in the backward constructor."""

    @staticmethod
    def forward(ctx, *input, **kwargs):
        """
        torch.autograd.Function's forward function has **kwargs to parameterize the forward+backward function pair. We created CUDAGraphModulePreviousLayerFunction to replay the graph captured in the constructor.
        So we only need to pass all the tensors as is to make sure the replay does executed by the PyTorch autograd engine. We do not need to pass **kwargs from the replayed function.
        In future, if we need to parameterize the CUDAGraphModulePreviousLayerFunction itself, we may add **kwargs.
        """
        ctx.backward_constructor = kwargs["backward_constructor"]
        ctx.save_for_backward(*input)
        return (*input,)

    @staticmethod
    def backward(ctx, *grad_tensor_input):
        backward_constructor = ctx.backward_constructor
        backward_constructor.instantiate_graph_exec()
        backward_constructor.execute_graph_exec()
        backward_constructor.synchronize()
        backward_constructor.destroy_graph_exec()
        return (None, *grad_tensor_input)


class CUDAGraphModuleNextLayerFunction(torch.autograd.Function):
    """The layer that replays the graph captured in the forward constructor."""

    @staticmethod
    def forward(ctx, *input, **kwargs):
        """
        torch.autograd.Function's forward function has **kwargs to parameterize the forward+backward function pair. We created CUDAGraphModuleNextLayerFunction to replay the graph captured in the constructor.
        So we only need to pass all the tensors as is to make sure the replay does executed by the PyTorch autograd engine. We do not need to pass **kwargs from the replayed function.
        In future, if we need to parameterize the CUDAGraphModuleNextLayerFunction itself, we may add **kwargs.
        """
        forward_constructor = kwargs["forward_constructor"]
        ctx.forward_constructor = forward_constructor
        ctx.save_for_backward(*input)
        forward_constructor.instantiate_graph()
        forward_constructor.execute_graph_exec()
        forward_constructor.synchronize()
        forward_constructor.destroy_graph_exec()
        return (*input,)

    @staticmethod
    def backward(ctx, *grad_tensor_input):
        return (None, *grad_tensor_input)


# TODO: define nn.Modules 1) SpMM + GEMM 2) FP16 + FP32
class ModuleToReplay(nn.Module):
    num_parallel_autograd_funcs: int
    layer_func: MyAutogradFunc
    ...

    def forward(
        self,
        *input_and_weights,
        forward_constructors: list[TorchCUDAGraphConstructor],
        backward_constructors: list[TorchCUDAGraphConstructor],
    ):
        raise NotImplementedError


class ConstructorEnabledLayer(nn.Module):
    def __init__(self, layer_to_replay: ModuleToReplay):
        super().__init__()
        # Initialize constructors
        self.layer_to_replay = layer_to_replay
        # autograd_func.register_constructor(
        #     backward_constructor, forward_constructor
        # )
        self.weights: list[torch.Tensor] = []

    def forward(self, x):
        backward_constructors = []
        forward_constructors = []
        for idx in range(self.layer_to_replay.num_parallel_autograd_funcs):
            backward_constructor = TorchCUDAGraphConstructor()
            forward_constructor = TorchCUDAGraphConstructor()
            backward_constructors.append(backward_constructor)
            forward_constructors.append(forward_constructor)
        x = CUDAGraphModulePreviousLayerFunction.apply(
            x, backward_constructor=backward_constructors[0]
        )
        # TODO: what arguments to pass to layer_to_replay?
        x = self.layer_to_replay(
            *x,
            *self.weights,
            forward_constructors=forward_constructors,
            backward_constructors=backward_constructors,
        )
        x = CUDAGraphModuleNextLayerFunction.apply(
            x, forward_constructor=forward_constructors[0]
        )
        return x
