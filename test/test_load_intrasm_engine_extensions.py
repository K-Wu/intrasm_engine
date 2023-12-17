if __name__ == "__main__":
    import intrasm_engine
    import intrasm_engine_extensions as iee

    print(iee.CUDAExperimentalGraphConstructor)

    import torch

    stream = torch.cuda.Stream()
    print("stream addr <python>: ", stream.cuda_stream)
    # stream.cuda_stream gets the value of cudaStream_t (which is a pointer) according to THCPStream_get_cuda_stream in pytorch/torch/csrc/cuda/Stream.cpp
    print(
        "stream addr round-trip <python>: ",
        iee.print_cudastream(stream.cuda_stream),
    )
