import torch
from cuda import cuda


def test_pycuda_cuda_torch_interop():
    import pycuda.autoinit  # Initialize pycuda
    import pycuda.driver

    pycuda_stream = pycuda.driver.Stream()
    custream = cuda.CUstream(init_value=pycuda_stream.handle)
    curesult, id = cuda.cuStreamGetId(custream)
    stream = torch.cuda.Stream(stream_ptr=pycuda_stream.handle)
    custream_2 = cuda.CUstream(init_value=stream.cuda_stream)
    curesult, id_2 = cuda.cuStreamGetId(custream_2)
    print("id: ", id)
    print("id_2: ", id_2)
    assert id == id_2

    # Test further if the stream is correctly produced
    torch.cuda.set_stream(stream)
    a = torch.randn(100, 100, device="cuda")
    b = torch.randn(100, 100, device="cuda")
    c = torch.matmul(a, b)


def test_cuda_torch_intrasm_engine_interop():
    from cuda import cudart

    stream = torch.cuda.Stream()
    # CUstream definition: https://github.com/NVIDIA/cuda-python/blob/dfd31fa609b9c81bcff925824f38531ab3c96706/cuda/cuda.pyx.in
    # void_ptr is unsigned long long as defined in https://github.com/NVIDIA/cuda-python/blob/dfd31fa609b9c81bcff925824f38531ab3c96706/cuda/_lib/utils.pyx.in#L20
    # Custream usage: https://github.com/NVIDIA/cutlass/blob/b7508e337938137a699e486d8997646980acfc58/python/cutlass/utils/profiler.py#L57
    custream = cuda.CUstream(init_value=stream.cuda_stream)

    event_start = torch.cuda.Event(enable_timing=True)
    event_start.record()
    event_stop = torch.cuda.Event(enable_timing=True)
    # Do a matrix multiply
    a = torch.randn(100, 100, device="cuda")
    b = torch.randn(100, 100, device="cuda")
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    event_stop.record()
    print("pytorch elapsed time: ", event_start.elapsed_time(event_stop))

    cuevent_start = cuda.CUevent(init_value=event_start.cuda_event)
    cuevent_stop = cuda.CUevent(init_value=event_stop.cuda_event)
    # Test if destroying CUevent will destroy the underlying cudaEvent_t
    del cuevent_start
    del cuevent_stop
    cuevent_start = cuda.CUevent(init_value=event_start.cuda_event)
    cuevent_stop = cuda.CUevent(init_value=event_stop.cuda_event)

    print("cuevent_start: ", cuevent_start)
    print("cuevent_stop: ", cuevent_stop)
    curesult, elapsed_time = cudart.cudaEventElapsedTime(
        cuevent_start, cuevent_stop
    )
    print("elapsed_time: ", elapsed_time)
    assert elapsed_time == event_start.elapsed_time(event_stop)


if __name__ == "__main__":
    test_cuda_torch_intrasm_engine_interop()
    test_pycuda_cuda_torch_interop()
