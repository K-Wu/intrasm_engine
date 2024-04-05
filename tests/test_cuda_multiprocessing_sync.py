"""
This process creates a new process and synchronize the two processes using a shared CUDA tensor using CUDAMultiprocessingSync instant.
The process creation scheme is adapted from https://github.com/davidmin7/dgl/blob/c96a8b3e91d0a6cdbb8b103fe84b1374e94053f9/examples/pytorch/graphsage/train_sampling_pytorch_direct.py
Specifically, MPS is used to set the percentage of SMs allocated to each process. The utility of MPS is at https://github.com/davidmin7/dgl/blob/c96a8b3e91d0a6cdbb8b103fe84b1374e94053f9/examples/pytorch/graphsage/utils.py
Best practices at https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors
"""
