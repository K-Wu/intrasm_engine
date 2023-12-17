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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("print_cudastream", &print_cudastream, "print cudaStream_t");
  // Data structures
  py::class_<CUDAExperimentalGraphConstructor<cudaStream_t>>(
      m, "CUDAExperimentalGraphConstructor")
      .def(py::init<>());
}
