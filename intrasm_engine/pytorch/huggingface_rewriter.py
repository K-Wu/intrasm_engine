# Based on the peft library that uses get_submodule method and setattr built-in method to find the layer and conduct the replacement.
# Reference: the answer at https://discuss.huggingface.co/t/how-can-i-replace-modules-in-a-pretrained-model/16338/4. An alternative and simpler solution is in the previous answer in the same thread.
# parent, target, target_name = _get_submodules(model, key)
from __future__ import annotations
from peft.utils.other import _get_submodules

# check_target_module_exists(config, key: str) -> bool | re.Match[str] | None
from peft.tuners.tuners_utils import check_target_module_exists
import re
import torch
from torch import nn
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers import TrainerCallback

from .cpp_extensions import cuda_graph_constructor

import logging

logger = logging.getLogger(__name__)


class IntraSMCallback(TrainerCallback):
    """Register the tensor dictionary and the config dictionary to which the inserted KernelPairModule refer for forward propagation and backward propagation.
    Usage: trainer = Trainer(..., callbacks=[IntreSMCallback(tensor_dict, config_dict)])
    or trainer.add_callback(IntreSMCallback(tensor_dict, config_dict))

    Based on https://github.com/microsoft/nni/blob/767ed7f22e1e588ce76cbbecb6c6a4a76a309805/nni/compression/utils/evaluator.py#L1076
    """

    def __init__(self, tensor_dict, config_dict):
        self.tensor_dict = tensor_dict
        self.config_dict = config_dict

    def on_train_begin(self, args, state, control, **kwargs):
        """Register the tensor dictionary and the config dictionary to which the inserted KernelPairModule refer for forward propagation and backward propagation."""
        state["tensor_dict"] = self.tensor_dict
        state["config_dict"] = self.config_dict

    def on_step_begin(self, args, state, control, **kwargs):
        """Register the tensor dictionary and the config dictionary to which the inserted KernelPairModule refer for forward propagation and backward propagation."""
        state["tensor_dict"] = self.tensor_dict
        state["config_dict"] = self.config_dict

    def on_step_end(self, args, state, control, **kwargs):
        """Register the tensor dictionary and the config dictionary to which the inserted KernelPairModule refer for forward propagation and backward propagation."""
        state["tensor_dict"] = self.tensor_dict
        state["config_dict"] = self.config_dict


class IntraSMAdapter(nn.Module):
    """
    Based on the peft.IA3Model adapter class.
    """

    def __init__(self, model, peft_config, adapter_name):
        super().__init__()

        self.model = model

        # For advanced developpers, if you want to attach multiple adapters to your
        # model, just add a `peft_config` dict attribute to your model.
        if not hasattr(self, "peft_config"):
            self.peft_config = (
                {adapter_name: peft_config}
                if isinstance(peft_config, PeftConfig)
                else peft_config
            )
        else:
            logger.info(
                "Already found a `peft_config` attribute in the model. This"
                " will lead to having multiple adapters in the model. Make"
                " sure to know what you are doing!"
            )
            if isinstance(peft_config, PeftConfig):
                self.peft_config[adapter_name] = peft_config
            else:
                # user is adding a dict of PeftConfigs
                self.peft_config.update(peft_config)

        self.active_adapter = adapter_name

        # transformers models have a .config attribute, whose presence is assumed later on
        if not hasattr(self, "config"):
            self.config = {"model_type": "custom"}

        self.inject_adapter(self.model, adapter_name)

        # Copy the peft_config in the injected model.
        self.model.peft_config = self.peft_config

    def inject_adapter(
        self, model: nn.Module | LlamaForCausalLM, adapter_name: str
    ):
        r"""
        Based on inject_adapter from https://github.com/huggingface/peft/blob/c0dd27bc974e4a62c6072142146887b75bb2de6c/src/peft/tuners/tuners_utils.py#L230-L234.

        Creates adapter layers and replaces the target modules with the adapter layers. This method is called under the
        hood by `peft.mapping.get_peft_model` if a non-prompt tuning adapter class is passed.

        The corresponding PEFT config is directly retrieved from the `peft_config` attribute of the BaseTuner class.

        Args:
            model (`nn.Module`):
                The model to be tuned.
            adapter_name (`str`):
                The adapter name.
        """
        peft_config = self.peft_config[adapter_name]
        # Note: If possible, all checks should be performed *at the start of this method*.
        # This way, we can raise early if something goes wrong, without leaving the model
        # in a bad (half-initialized) state.
        self._check_new_adapter_config(peft_config)

        is_target_modules_in_base_model = False
        key_list = [key for key, _ in model.named_modules()]

        model_config = getattr(model, "config", {"model_type": "custom"})
        if hasattr(model_config, "to_dict"):
            model_config = model_config.to_dict()

        peft_config = self._prepare_adapter_config(peft_config, model_config)

        for key in key_list:
            # Skipping the "Check for modules_to_save in case"step

            if not check_target_module_exists(peft_config, key):
                continue

            is_target_modules_in_base_model = True
            try:
                parent, target, target_name = _get_submodules(model, key)
            except AttributeError as e:
                # the module was replaced in previous iteration and no longer in the model. Skipping it
                logger.warning(
                    f"Skipping module {key} as it was replaced in previous"
                    " iteration and no longer in the model."
                )
                continue

            optional_kwargs = {
                "loaded_in_8bit": getattr(model, "is_loaded_in_8bit", False),
                "loaded_in_4bit": getattr(model, "is_loaded_in_4bit", False),
                "current_key": key,
            }
            self._create_and_replace(
                peft_config,
                adapter_name,
                target,
                target_name,
                parent,
                **optional_kwargs,
            )

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {peft_config.target_modules} not found in the"
                " base model. Please check the target modules and try again."
            )

        # Mark adapters as trainable
        for n, p in model.named_parameters():
            if self.prefix in n:
                p.requires_grad = True

        if self.peft_config[adapter_name].inference_mode:
            for n, p in self.model.named_parameters():
                if adapter_name in n:
                    p.requires_grad = False

        # Skipping the "Saving modules_to_save" step as the checking step that bookkeep the modules_to_save is also skipped

    def _check_new_adapter_config(self, config: Configs) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        return

    def _create_and_replace(
        self,
        ia3_config,
        adapter_name,
        target,
        target_name,
        parent,
        **optional_kwargs,
    ):
        """
        Based on _create_and_replace at https://github.com/huggingface/peft/blob/cf04d0353f0343cbf66627228c4495f51669af34/src/peft/tuners/ia3/model.py.
        """
        loaded_in_8bit = optional_kwargs["loaded_in_8bit"]
        loaded_in_4bit = optional_kwargs["loaded_in_4bit"]
        current_key = optional_kwargs["current_key"]

        # Skip the "check if target module is in feedforward_modules" logic

        kwargs = {
            "fan_in_fan_out": ia3_config.fan_in_fan_out,
            "init_ia3_weights": ia3_config.init_ia3_weights,
            "loaded_in_8bit": loaded_in_8bit,
            "loaded_in_4bit": loaded_in_4bit,
        }

        # Skip the "if target is the demanded layer type, update the layer instead of create a new one" logic
        assert not isinstance(target, IA3Layer)
        new_module = self._create_new_module(
            ia3_config, adapter_name, target, **kwargs
        )
        # Skip the "if the adapter is inactive, set the adapter to be not trainable" logic
        self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(ia3_config, adapter_name, target, **kwargs):
        # avoid eager bnb import
        if is_bnb_available():
            import bitsandbytes as bnb

            from .bnb import Linear8bitLt

        if is_bnb_4bit_available():
            from .bnb import Linear4bit

        loaded_in_8bit = kwargs.pop("loaded_in_8bit", False)
        loaded_in_4bit = kwargs.pop("loaded_in_4bit", False)
        is_feedforward = kwargs.pop("is_feedforward", False)

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if loaded_in_8bit and isinstance(
            target_base_layer, bnb.nn.Linear8bitLt
        ):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target_base_layer.state.has_fp16_weights,
                    "memory_efficient_backward": target_base_layer.state.memory_efficient_backward,
                    "threshold": target_base_layer.state.threshold,
                    "index": target_base_layer.index,
                }
            )
            new_module = Linear8bitLt(
                target,
                adapter_name,
                is_feedforward=is_feedforward,
                **eightbit_kwargs,
            )
        elif loaded_in_4bit and isinstance(
            target_base_layer, bnb.nn.Linear4bit
        ):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target_base_layer.compute_dtype,
                    "compress_statistics": target_base_layer.weight.compress_statistics,
                    "quant_type": target_base_layer.weight.quant_type,
                }
            )
            new_module = Linear4bit(
                target,
                adapter_name,
                is_feedforward=is_feedforward,
                **fourbit_kwargs,
            )
        elif isinstance(target, torch.nn.Conv2d):
            new_module = Conv2d(
                target, adapter_name, is_feedforward=is_feedforward, **kwargs
            )
        elif isinstance(target_base_layer, torch.nn.Linear):
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is"
                    " `torch.nn.Linear`. Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = ia3_config.fan_in_fan_out = False
            new_module = Linear(
                target, adapter_name, is_feedforward=is_feedforward, **kwargs
            )
        elif isinstance(target_base_layer, Conv1D):
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is"
                    " `Conv1D`. Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = ia3_config.fan_in_fan_out = True
            new_module = Linear(
                target,
                adapter_name,
                is_feedforward=is_feedforward,
                is_target_conv_1d_layer=True,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only"
                " `torch.nn.Linear`, `torch.nn.Conv2d`, and `Conv1D` are"
                " supported."
            )
        return new_module

    def _replace_module(self, parent, child_name, new_module, child):
        """
        Based on _replace_module at https://github.com/huggingface/peft/blob/cf04d0353f0343cbf66627228c4495f51669af34/src/peft/tuners/ia3/model.py.
        The base code is a superset of another example at https://github.com/huggingface/peft/blob/cf04d0353f0343cbf66627228c4495f51669af34/src/peft/tuners/lycoris_utils.py, which does not unpack the child layer if applicable as the first step.
        """
        setattr(parent, child_name, new_module)

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        # layers with base_layer don't need the weight to be copied, as they have a reference already
        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if self.prefix in name:
                module.to(child.weight.device)
