// From
// NVIDIA/TransformerEngine/transformer_engine/pytorch/csrc/extensions/pybind.cpp
#include <cuda_runtime.h>

#include <iostream>

#include "../extensions.h"

static long print_cudastream(long st) {
  void* st_v = (void*)st;
  cudaStream_t stream = static_cast<cudaStream_t>(st_v);
  std::cout << "stream: " << stream << std::endl;
  return PyLong_AsLong(PyLong_FromVoidPtr(st_v));
}

static long print_cudaevent(long st) {
  void* st_v = (void*)st;
  if (st_v == nullptr) {
    throw std::runtime_error(
        "torch.cuda.event is nullptr. Please initialize it first. Notice that "
        "currently the event is only initialized after record() is called.");
  }
  cudaEvent_t event = static_cast<cudaEvent_t>(st_v);
  std::cout << "event: " << event << std::endl;
  return PyLong_AsLong(PyLong_FromVoidPtr(st_v));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("print_cudastream", &print_cudastream, "print cudaStream_t");
  m.def("print_cudaevent", &print_cudaevent, "print cudaEvent_t");
  // Data structures
  py::class_<CUDAExperimentalGraphConstructor<cudaStream_t>>(
      m, "CUDAExperimentalGraphConstructor")
      .def(py::init<>());
  //  .def("register_stream",
  //       &CUDAExperimentalGraphConstructor<cudaStream_t>::registerStream)
  //  .def("register_stream_last_activity",
  //       &CUDAExperimentalGraphConstructor<
  //           cudaStream_t>::registeStreamLastActivity)
  //  .def("add_event_record_node",
  //       &CUDAExperimentalGraphConstructor<cudaStream_t>::addEventRecordNode)
  //  .def("add_stream_wait_event_node",
  //       &CUDAExperimentalGraphConstructor<
  //           cudaStream_t>::addStreamWaitEventNode)
  //  .def("join", &CUDAExperimentalGraphConstructor<cudaStream_t>::join)
  //  .def("notify_before_invoking_library_call",
  //       &CUDAExperimentalGraphConstructor<
  //           cudaStream_t>::notifyBeforeInvokingLibraryCall)
  //  .def("notify_after_invoking_library_call",
  //       &CUDAExperimentalGraphConstructor<
  //           cudaStream_t>::notifyAfterInvokingLibraryCall);
}
