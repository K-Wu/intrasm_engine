from typing import Any, Callable, Optional
import torch
from sparta.common.tuning import TunableItemCfg
from sparta.nn.module_tuner import GridSearchTuner, RandomSearchTuner
import numpy as np
import logging
from ..pytorch.cpp_extensions.layers_and_funcs_utils import MyAutogradFunc

_logger = logging.Logger(__name__)
_handler = logging.StreamHandler()
_logger.addHandler(_handler)


def tune_my_autograd_function(
    autograd_func: type[MyAutogradFunc],
    name: str,
    sample_inputs: list[torch.Tensor],
    sample_grads: Optional[list[torch.Tensor]] = None,
    algo: str = "grid",
    max_trials: int = 40,
):
    """Adapted from tune_sparse_module in https://github.com/K-Wu/SparTA/blob/1a0a0b604979d158ef016e2b9f43705bbb9c55e0/sparta/nn/module_tuner.py"""
    if algo.startswith("grid"):
        tuner_type = GridSearchTuner
    elif algo.startswith("rand"):
        tuner_type = RandomSearchTuner
    else:
        raise ValueError(f'unsupported tuner algorithm "{algo}"')

    upper_space = autograd_func.get_search_space(sample_inputs, sample_grads)

    upper_space_shape = [
        len(upper_space[param_name]._value) for param_name in upper_space
    ]
    upper_space_size = int(np.prod(upper_space_shape))

    def lower_search(
        tuner_space_idx: int, tuner_chosen_params: dict[Any, Any]
    ):
        _logger.info(
            f"[{name}][Upper Search Space] #{tuner_space_idx}:"
            f" {list(tuner_chosen_params.values())}"
        )
        _logger.info(
            f"[{name}][Kernel:"
            f"{str(autograd_func)} {[input.shape for input in sample_inputs]} "
            f"{[grad.shape for grad in sample_grads] if sample_grads is not None else ''}]"
        )

        kernel_latency = autograd_func.build_and_test(
            sample_inputs, sample_grads, tuner_chosen_params
        )

        return kernel_latency

    tuner = tuner_type(
        search_space=upper_space,
        eval_func=lower_search,
        max_trials=min(max_trials, upper_space_size),
    )
    tuner.tune()
    _logger.info(f"[{name}] Tuning completed.")
    if tuner.best_config is None:
        _logger.warn(f"[{name}] All trials failed.")
        return None
    else:
        _logger.info(f"[{name}] Best config:\n{tuner.best_config}")
        return tuner.best_config
