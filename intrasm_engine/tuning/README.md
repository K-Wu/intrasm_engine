## Overview
The tuning mechanism uses nni to tune the parameters once the search space is defined. We use the data structure in [Microsoft/SparTA/sparta/common/tuning.py - Github](https://github.com/K-Wu/SparTA/blob/1a0a0b604979d158ef016e2b9f43705bbb9c55e0/sparta/common/tuning.py) to help us define the search space.

## TODO: Plan to incorporate xgboost
We may incorporate xgboost to model the cost. An example is in [DeepSpeed](https://github.com/microsoft/DeepSpeed/blob/16c265c0ce103147d027d9cae32dd7680766af21/deepspeed/autotuning/tuner/cost_model.py).

## Potential Reference on GPU Performance Modeling
Souley Madougou. The landscape of GPGPU performance modeling tools.

Ying Zhang et al. Performance and Power Analysis of ATI GPU: A Statistical Approach.