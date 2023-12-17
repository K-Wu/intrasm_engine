if __name__ == "__main__":
    import intrasm_engine
    import intrasm_engine_extensions as iee

    print(iee.CUDAExperimentalGraphConstructor)

    import torch

    stream = torch.cuda.Stream()
    print("stream addr <python>: ", stream.cuda_stream)
    print(
        "stream addr round-trip <python>: ",
        iee.print_cudastream(stream.cuda_stream),
    )
