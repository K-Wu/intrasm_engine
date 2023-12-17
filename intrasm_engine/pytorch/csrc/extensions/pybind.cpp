// From
// NVIDIA/TransformerEngine/transformer_engine/pytorch/csrc/extensions/pybind.cpp

#include "../extensions.h"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Data structures
  py::class_<CUDAExperimentalGraphConstructor<cudaStream_t>>(
      m, "CUDAExperimentalGraphConstructor")
      .def(py::init<>());
}
