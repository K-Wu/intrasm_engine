"""
This process creates a new process and synchronize the two processes using a shared CUDA tensor using CUDAMultiprocessingSync instant.
The process creation scheme is adapted from https://github.com/davidmin7/dgl/blob/c96a8b3e91d0a6cdbb8b103fe84b1374e94053f9/examples/pytorch/graphsage/train_sampling_pytorch_direct.py
Spawn is used in creating the new process to make CUDA tensors work with multiple processes. See https://stackoverflow.com/questions/50735493/how-to-share-a-list-of-tensors-in-pytorch-multiprocessing.
Specifically, MPS is used to set the percentage of SMs allocated to each process. The utility of MPS is at https://github.com/davidmin7/dgl/blob/c96a8b3e91d0a6cdbb8b103fe84b1374e94053f9/examples/pytorch/graphsage/utils.py
PyTorch has the reference counting mechanism support for multi-processing under the hood, but it is still necessary to follow the best practices. See https://github.com/pytorch/pytorch/blob/main/torch/multiprocessing/cuda_multiprocessing.md, and https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors
"""
