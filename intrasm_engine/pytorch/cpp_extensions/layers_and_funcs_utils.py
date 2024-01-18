from typing import Optional, Any
import torch
from sparta.common.tuning import TunableItemCfg


class MyAutogradFunc(torch.autograd.Function):
    @staticmethod
    def num_inputs(**fwd_kwargs_unwrapped) -> int:
        raise NotImplementedError(
            "You must implement num_inputs for your autograd function."
        )

    @staticmethod
    def num_outputs(**ctx_kwargs_unwrapped) -> int:
        raise NotImplementedError(
            "You must implement num_outputs for your autograd function."
        )

    # @staticmethod
    # def num_input_tensors(**fwd_kwargs_unwrapped) -> int:
    #     raise NotImplementedError(
    #         "You must implement num_input_tensors for your autograd function."
    #     )

    @staticmethod
    def num_saved_tensors(**ctx_kwargs_unwrapped) -> int:
        raise NotImplementedError(
            "You must implement num_saved_tensors for your autograd function."
        )

    @staticmethod
    def _forward(
        *args, **fwd_kwargs_unwrapped
    ) -> tuple[tuple[torch.Tensor, ...], dict, tuple[torch.Tensor, ...]]:
        """
        This method returns a tuple of outputs, dictionary to save in ctx, and tensors_to_save for backward.
        We do not save_for_backward on the tensors because save_for_backward can only be executed once.
        In the KernelPairOrTripleAutogradFunction, we have to execute _forward twice/thrice. Our solution is to collect the tensors to save in the two/three _forward() function calls, and save the tensors only once in the forward() function.
        """
        raise NotImplementedError(
            "You must implement _forward for your autograd function."
        )

    @staticmethod
    def _backward(
        *grad_and_saved_tensors, **ctx_kwargs_unwrapped
    ) -> tuple[torch.Tensor, ...]:
        """
        This method actually takes in *args, *tensors_to_save, **kwargs.
        In the KernelPairOrTripleAutogradFunction, the saved tensor is a combination of the tensors_to_save from the two/three _forward() function calls. We have to unpack them in the backward() function, and pass them to _backward() function.
        """
        raise NotImplementedError(
            "You must implement _backward for your autograd function."
        )

    @classmethod
    def get_search_space(
        cls,
        sample_inputs: list[torch.Tensor],
        sample_grads: Optional[list[torch.Tensor]],
    ) -> dict[str, TunableItemCfg]:  # {'param_name': {param_values}}
        raise NotImplementedError(
            "You must implement get_search_space for your autograd function."
        )

    @classmethod
    def check_params_sanity(cls, params: dict):
        raise NotImplementedError(
            "You must implement check_params_sanity for your autograd"
            " function."
        )

    @classmethod
    def check_fwd_kwargs_sanity(cls, fwd_kwargs: dict[str, Any]):
        assert "fwd_kwargs_1" in fwd_kwargs
        raise NotImplementedError(
            "You must implement check_fwd_kwargs_sanity for your"
            " autograd function."
        )

    @classmethod
    def check_ctx_dict_sanity(cls, ctx_dict: dict[str, Any]):
        raise NotImplementedError(
            "You must implement check_ctx_dict_sanity for your autograd"
            " function."
        )

    @classmethod
    def get_test_fwd_kwargs(cls, params: dict) -> dict[str, Any]:
        raise NotImplementedError(
            "You must implement get_test_fwd_kwargs for your autograd"
            " function."
        )

    @classmethod
    def build_and_test(
        cls,
        sample_inputs: list[torch.Tensor],
        sample_grads: Optional[list[torch.Tensor]],
        params: dict,
    ):
        fwd_kwargs_unwrapped = cls.get_test_fwd_kwargs(params)["fwd_kwargs_1"]
        outputs, ctx_dict_to_save, tensors_to_save = cls._forward(
            *sample_inputs, **fwd_kwargs_unwrapped
        )
        if sample_grads is None:
            sample_grads = tuple(
                torch.randn_like(output) for output in outputs
            )
        assert sample_grads is not None  # Suppress type checker error
        grad_inputs = cls._backward(
            *sample_grads, *tensors_to_save, **ctx_dict_to_save
        )
        # TODO: Replay forward and backward graph constructors and get timing
        assert fwd_kwargs_unwrapped["constructor_enabled"]
