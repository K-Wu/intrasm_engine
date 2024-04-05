# Intra-Streaming-Multiprocessor (IntraSM) Engine


## Contributing
### Setuptools Development Mode
This repository uses setup.py to build the package. To develop, install the package in editable mode:
```
pip install -e .
```

[Python setup.py develop vs install - StackOverflow](https://stackoverflow.com/a/19048754)

## Installation
We need [scikit-sparse](https://github.com/scikit-sparse/scikit-sparse/tree/master) in some data-exploring and benchmark code. It needs additional step to pip install. Check the repo page for more details.

### Pybind Overloading
[Binding Disambiguation - pybind11 Documentation](https://pybind11.readthedocs.io/en/stable/classes.html#:~:text=We%20can%20disambiguate%20by%20casting%20them%20to%20function%20pointers)

[Adding Lambda Function as a Class Method - pybind11 Documentation](https://pybind11.readthedocs.io/en/stable/classes.html#:~:text=Unfortunately%2C%20there%20is%20no%20suitable%20functionality%20in%20the%20Pet%20data%20structure%2C%20and%20it%20would%20be%20nice%20if%20we%20did%20not%20have%20to%20change%20it.)

### Pybind Resource Management
[Smart Pointers](https://pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html)

### Directory Structure
This repository follows the directory structure of [TransformerEngine - Github](https://github.com/NVIDIA/TransformerEngine/).

### Profiling
[yqhu/profiler-workshop - Github](https://github.com/yqhu/profiler-workshop) provides examples on using PyTorch profiler (in Huggingface models) to profile the model.

1. [hf_pipeline_prof.py](https://github.com/yqhu/profiler-workshop/blob/c8d4a7c30a61cc7b909d89f88f5fd36b70c55769/hf_pipeline_prof.py) demonstrates how to export the profiling results as json traces and FlameGraph.
2. [hf_training_trainer_prof.py](https://github.com/yqhu/profiler-workshop/blob/c8d4a7c30a61cc7b909d89f88f5fd36b70c55769/hf_training_trainer_prof.py) demonstrates how to profile a huggingface model via registering TrainerCallback.
3. [hf_training_torch_prof.py](https://github.com/yqhu/profiler-workshop/blob/c8d4a7c30a61cc7b909d89f88f5fd36b70c55769/hf_training_torch_prof.py) demonstrates how to run the Huggingface model in steps and profile it via PyTorch profiler in native manner.

#### Nsight Compute Flags
Consider using the following to obtain Nsight Compute profiling results where inter-kernel interference is recorded.
```
--cache-control none --replay-mode application
```

## Code Health Badges
[![CodeFactor](https://www.codefactor.io/repository/github/k-wu/intrasm_engine/badge?s=749489c3b14056d2ece1446c9f6f3e55572069b3)](https://www.codefactor.io/repository/github/k-wu/intrasm_engine)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/efbb131ba609458c8a586ea63c2534e2)](https://app.codacy.com?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![DeepSource](https://app.deepsource.com/gh/K-Wu/intrasm_engine.svg/?label=active+issues&show_trend=true&token=OE3XZsUS8QPEMWILgPiJbtGG)](https://app.deepsource.com/gh/K-Wu/intrasm_engine/)

## Library Supports
### Multistream
We have incorporated stream switch support in [our custom SparTa repo](https://github.com/K-Wu/SparTA).

For Cutlass, the Python interface does have stream support, but it is not exposed to the top-level API. We filed a [PR](https://github.com/NVIDIA/cutlass/pull/1287), now merged, to expose the stream support to the top-level API.

### CUDA Library Determinism
We don't preserve cuBLAS determinism for now.

cuBLAS has the determinism issue, which requires either 1) one handle per stream or 2) one workspace per stream. In cupy, no workspace setting API is exposed, and each device got a default handle. We also need to check if there is any necessary additional handling in PyTorch to guarantee determinism.

No reported issue about cusparse determinism, and I guess the reason is it is deterministic because it has a specific bufferSize allocation operation for each SpMM operation.

### Reference
[Is a pool of cuBLAS handles required for stream parallelism? #4676](https://github.com/cupy/cupy/issues/4676)
[cuBLAS reproducibility](https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility)
[cuSparse documentation](https://docs.nvidia.com/cuda/cusparse)

## Auto-tuning
### Microbenchmark
We use the [Accel-Sim microbenchmark suites](https://github.com/accel-sim/gpu-app-collection/blob/release/src/cuda/GPU_Microbenchmark/), which is based on ["Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking"](https://arxiv.org/pdf/1804.06826.pdf)

## Contact
Kun Wu kunwu2 (at) illinois (dot) edu  [![wakatime](https://wakatime.com/badge/github/K-Wu/intrasm_engine.svg)](https://wakatime.com/badge/github/K-Wu/intrasm_engine)