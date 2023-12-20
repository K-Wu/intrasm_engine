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

static std::vector<long> print_cudastreams(std::vector<long> sts) {
  std::vector<long> ret;
  std::cout << "streams: " << std::endl;
  for (auto st : sts) {
    ret.push_back(print_cudastream(st));
  }
  return ret;
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

static void PyWrapper_registerStream(
    CUDAExperimentalGraphConstructor<cudaStream_t>& graph, long stream) {
  void* stream_v = (void*)stream;
  cudaStream_t stream_t = static_cast<cudaStream_t>(stream_v);
  graph.registerStream(stream_t);
}

static void PyWrapper_addEventRecordNode(
    CUDAExperimentalGraphConstructor<cudaStream_t>& graph, long event,
    long stream) {
  void* event_v = (void*)event;
  cudaEvent_t event_t = static_cast<cudaEvent_t>(event_v);
  void* stream_v = (void*)stream;
  cudaStream_t stream_t = static_cast<cudaStream_t>(stream_v);
  graph.addEventRecordNode(event_t, stream_t);
}

static void PyWrapper_addStreamWaitEventNode(
    CUDAExperimentalGraphConstructor<cudaStream_t>& graph, long stream,
    long event) {
  void* stream_v = (void*)stream;
  cudaStream_t stream_t = static_cast<cudaStream_t>(stream_v);
  void* event_v = (void*)event;
  cudaEvent_t event_t = static_cast<cudaEvent_t>(event_v);
  graph.addStreamWaitEventNode(stream_t, event_t);
}

static void PyWrapper_join(
    CUDAExperimentalGraphConstructor<cudaStream_t>& graph,
    std::vector<long> streams, long dst_stream) {
  std::vector<cudaStream_t> streams_t;
  for (auto stream : streams) {
    void* stream_v = (void*)stream;
    cudaStream_t stream_t = static_cast<cudaStream_t>(stream_v);
    streams_t.push_back(stream_t);
  }
  void* dst_stream_v = (void*)dst_stream;
  cudaStream_t dst_stream_t = static_cast<cudaStream_t>(dst_stream_v);
  graph.join(streams_t, dst_stream_t);
}

static void PyWrapper_notifyBeforeInvokingLibraryCall(
    CUDAExperimentalGraphConstructor<cudaStream_t>& graph, long stream) {
  void* stream_v = (void*)stream;
  cudaStream_t stream_t = static_cast<cudaStream_t>(stream_v);
  graph.notifyBeforeInvokingLibraryCall(stream_t);
}

static void PyWrapper_notifyAfterInvokingLibraryCall(
    CUDAExperimentalGraphConstructor<cudaStream_t>& graph, long stream) {
  void* stream_v = (void*)stream;
  cudaStream_t stream_t = static_cast<cudaStream_t>(stream_v);
  graph.notifyAfterInvokingLibraryCall(stream_t);
}

static long PyWrapper_combineGraphs(
    CUDAExperimentalGraphConstructor<cudaStream_t>& graph_constructor_lhs,
    CUDAExperimentalGraphConstructor<cudaStream_t>& graph_constructor_rhs) {
  // Merge the two graphs and then launch it
  cudaGraph_t merged_graph =
      cudaGraphCreateCombinedGraph(std::vector<cudaGraph_t>{
          graph_constructor_lhs.getGraph(), graph_constructor_rhs.getGraph()});
  // Avoid double freeing the child graph, i.e., the graph of the GEMM
  // partitioned since the destruction of parent graph will destroy it already
  graph_constructor_lhs.getGraphWrapper()->notifyAddedAsChildGraph();
  graph_constructor_rhs.getGraphWrapper()->notifyAddedAsChildGraph();
  return PyLong_AsLong(PyLong_FromVoidPtr(merged_graph));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("print_cudastream", &print_cudastream, "print cudaStream_t");
  m.def("print_cudastream", &print_cudastreams,
        "print multiple cudaStream_t (overloaded API)");
  m.def("print_cudaevent", &print_cudaevent, "print cudaEvent_t");
  // Data structures
  py::class_<CUDAExperimentalGraphConstructor<cudaStream_t>>(
      m, "CUDAExperimentalGraphConstructor")
      .def(py::init<>())
      .def("register_stream", &PyWrapper_registerStream)
      .def("add_event_record_node", &PyWrapper_addEventRecordNode)
      .def("add_stream_wait_event_node", &PyWrapper_addStreamWaitEventNode)
      .def("join", &PyWrapper_join)
      .def("notify_before_invoking_library_call",
           &PyWrapper_notifyBeforeInvokingLibraryCall)
      .def("notify_after_invoking_library_call",
           &PyWrapper_notifyAfterInvokingLibraryCall);
}
