import torch
import intrasm_engine
import intrasm_engine_extensions as iex


class CUDAMultiprocessingSync:
    """
    TODO: if we are to implement non-binary semaphore in the future, refer to
    https://stackoverflow.com/questions/31776702/standard-atomic-operations-of-semaphores
    When the address is passed to another process, CUDA IPC API should be used, which is what PyTorch used in share_memory_()
    https://stackoverflow.com/questions/19819216/interprocess-communication-cuda/19821387#19821387
    """

    def __init__(self, atomic_var: torch.Tensor | None):
        if atomic_var is None:
            atomic_var = torch.zeros(1, dtype=torch.int32, device="cuda")
        else:
            self.atomic_var = atomic_var

    def share_memory_(self):
        new_atomic_var = self.atomic_var.share_memory_()
        return CUDAMultiprocessingSync(new_atomic_var)

    def signal(self):
        iex.CUDAMultiprocessingSync_signal(self.atomic_var)

    def wait(self):
        iex.CUDAMultiprocessingSync_wait(self.atomic_var)
