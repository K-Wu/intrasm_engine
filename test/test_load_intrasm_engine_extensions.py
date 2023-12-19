if __name__ == "__main__":
    import intrasm_engine
    import intrasm_engine_extensions as iee  # Loading iee along without intrasm_engine will trigger libc10.so not found error

    print(iee.CUDAExperimentalGraphConstructor)

    import torch

    stream = torch.cuda.Stream()
    print("stream addr <python>: ", hex(stream.cuda_stream))
    # stream.cuda_stream gets the value of cudaStream_t (which is a pointer) according to THCPStream_get_cuda_stream in pytorch/torch/csrc/cuda/Stream.cpp
    print(
        "stream addr round-trip <python>: ",
        hex(iee.print_cudastream(stream.cuda_stream)),
    )

    event = torch.cuda.Event()
    event.record()
    print("event addr <python>: ", hex(event.cuda_event))
    # event.cuda_event gets the value of cudaEvent_t (which is a pointer) according to THCPEvent_get_cuda_event in pytorch/torch/csrc/cuda/Event.cpp
    print(
        "event addr round-trip <python>: ",
        hex(iee.print_cudaevent(event.cuda_event)),
    )
