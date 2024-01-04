// From
// NVIDIA/TransformerEngine/transformer_engine/pytorch/csrc/extensions/pybind.cpp
#include <cuda_runtime.h>

#include <iostream>

#include "../extensions.h"

// Define CUDAGraphConstructor
#include <helper_CUDAGraphConstructor.cu.h>
#include <helper_cuda_graph.cu.h>

namespace IntraSMEngine {

namespace {
/// Debug utilities
long print_cudastream(long st) {
  // This two-step casting is preferred:
  // https://stackoverflow.com/a/68137312/5555077
  void* st_v = reinterpret_cast<void*>(st);
  cudaStream_t stream = static_cast<cudaStream_t>(st_v);
  std::cout << "stream: " << stream << std::endl;
  return PyLong_AsLong(PyLong_FromVoidPtr(st_v));
}

std::vector<long> print_cudastreams(std::vector<long> sts) {
  std::vector<long> ret;
  std::cout << "streams: " << std::endl;
  for (auto st : sts) {
    ret.push_back(print_cudastream(st));
  }
  return ret;
}

long print_cudaevent(long st) {
  void* st_v = reinterpret_cast<void*>(st);
  if (st_v == nullptr) {
    throw std::runtime_error(
        "torch.cuda.event is nullptr. Please initialize it first. Notice that "
        "currently the event is only initialized after record() is called.");
  }
  cudaEvent_t event = static_cast<cudaEvent_t>(st_v);
  std::cout << "event: " << event << std::endl;
  return PyLong_AsLong(PyLong_FromVoidPtr(st_v));
}

/// CUDA Graph Constructor initialization and bookkeeping code
/// Based on
/// https://github.com/pytorch/pytorch/blob/ff4aac109a990e64d82fb73b5af5fa5e69278580/c10/cuda/CUDAStream.cpp
// Thread-local current CUDA Graph Constructors
thread_local std::vector<
    std::shared_ptr<CUDAExperimentalGraphConstructor<cudaStream_t>>>
    current_constructors{};

// Init front-end to ensure initialization only occurs once
void initCUDAGraphConstructorOnce() {
  if (current_constructors.size() > 0) {
    return;
  }

  // Inits current streams (thread local) to default streams
  for (const auto i : c10::irange(c10::cuda::device_count())) {
    current_constructors.emplace_back(
        std::make_shared<CUDAExperimentalGraphConstructor<cudaStream_t>>());
  }
}

// Helper to verify the GPU index is valid
inline void check_gpu(c10::DeviceIndex device_index) {
  TORCH_INTERNAL_ASSERT(device_index >= 0 &&
                        device_index < c10::cuda::device_count());
}

std::shared_ptr<CUDAExperimentalGraphConstructor<cudaStream_t>>
getCurrentGraphConstructor(c10::DeviceIndex device_index) {
  initCUDAGraphConstructorOnce();
  if (device_index == -1) {
    device_index = c10::cuda::current_device();
    c10::cuda::SetTargetDevice();
  }
  check_gpu(device_index);
  return current_constructors[device_index];
}

void setCurrentGraphConstructor(
    c10::cuda::CUDAStream stream,
    std::shared_ptr<CUDAExperimentalGraphConstructor<cudaStream_t>>
        constructor) {
  initCUDAGraphConstructorOnce();
  current_constructors[stream.device_index()] = constructor;
}

// Get the expected id of a capture sequence so that we can call
// beginAllocateStreamToPool before starting a graph capture
c10::cuda::CaptureId_t capture_sequence_id() {
  // id starts at 1:
  // Ensures uuid count starts at 1. 0 is reserved to mean "not set by
  // cudaStreamGetCaptureInfo". (But how do we know GetCaptureInfo never sets
  // id_ to 0? Because that's the current behavior, and I asked cuda devs to
  // keep it that way, and they agreed.)
  static std::atomic<c10::cuda::CaptureId_t> uuid{1};
  return uuid++;
}

/// CUDA Graph Capture Notifier Class
/// This is to bookkeep the pytorch states while using our CUDA Graph
/// Constructor.
/// Based on at::cuda::CUDAGraph at
/// https://github.com/pytorch/pytorch/blob/ff4aac109a990e64d82fb73b5af5fa5e69278580/c10/cuda/CUDAStream.cpp

// No need to inherit from torch::CustomClassHolder because the holder type is
// std::shared_ptr instead of c10::intrusive_ptr
class CUDAGraphCaptureNotifier {
 public:
  explicit CUDAGraphCaptureNotifier()
      : capture_stream_(at::cuda::getCurrentCUDAStream()) {
#if defined(USE_ROCM)
    TORCH_CHECK(false,
                "CUDAGraphCaptureNotifier is not supported on ROCm yet.");
#endif
  }

  void capture_begin(c10::cuda::MempoolId_t pool /*=0*/) {
    // The API and implementation of this function is based on
    // https://pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html

    // For now, a CUDAGraph instance only accommodates the default generator on
    // the device that's
    // current when capture begins. If any op in the captured region uses a
    // non-default generator, or a generator on another device, the offending
    // generator will throw an error. These restrictions simplify CUDAGraph, but
    // could be relaxed in the future: in principle, the underlying Cuda calls
    // do permit cross-device ops to be captured.
    auto* gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());

    auto options = c10::TensorOptions().device(at::kCUDA).dtype(at::kLong);
    seed_extragraph_ = at::empty({1}, options);
    offset_extragraph_ = at::empty({1}, options);

    seed_extragraph_.fill_(int64_t(gen->current_seed()));
    gen->capture_prologue(seed_extragraph_.data_ptr<int64_t>(),
                          offset_extragraph_.mutable_data_ptr<int64_t>());

    auto stream = at::cuda::getCurrentCUDAStream();

    TORCH_CHECK(stream != at::cuda::getDefaultCUDAStream(),
                "CUDA graphs must be captured on a non-default stream. "
                "(However, after capture, it's ok to replay them on the "
                "default stream.)");

    capture_stream_ = stream;
    capture_gen_ = gen;
    capture_dev_ = c10::cuda::current_device();

    id_ = capture_sequence_id();

    if (pool.first != 0 || pool.second != 0) {
      // Either value being nonzero means the user supplied a pool to share.
      // But only one should be nonzero.
      // If pool was created by another graph's capture_begin, first should be
      // nonzero. If pool was created by graph_pool_handle, second should be
      // nonzero.
      TORCH_INTERNAL_ASSERT(!(pool.first && pool.second));
      mempool_id_ = pool;
    } else {
      // User did not ask us to share a mempool. Use our own id_ as our
      // mempool_id_. Sets just the first value, to distinguish it from
      // MempoolId_ts created by graph_pool_handle().
      mempool_id_ = {id_, 0};
    }

    // Addendum: beginAllocateStreamToPool is now called before
    // cudaStreamBeginCapture to prevent an
    // autograd thread's free() call triggering an invalid cudaEventRecord in
    // the caching allocator due to the capture status being updated _after_ a
    // capture had already started.
    c10::cuda::CUDACachingAllocator::beginAllocateStreamToPool(
        capture_dev_, capture_stream_, mempool_id_);
  }

  void assert_capture_has_begun() {
    cudaStreamCaptureStatus status;
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, &capture_id_));
    TORCH_INTERNAL_ASSERT(
        status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive);
    TORCH_INTERNAL_ASSERT(id_ > 0);
  }

  void capture_end() {
    // The API and implementation of this function is based on
    // https://pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html

    c10::cuda::CUDACachingAllocator::endAllocateStreamToPool(capture_dev_,
                                                             capture_stream_);

    auto stream = at::cuda::getCurrentCUDAStream();

    TORCH_CHECK(stream == capture_stream_,
                "Capture must end on the same stream it began on.");
    auto* gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
    TORCH_CHECK(gen == capture_gen_,
                "Default CUDA RNG generator on current device at capture end "
                "is different from default generator on current device "
                "when capture began");
    wholegraph_increment_ = gen->capture_epilogue();
  }

  void replay() {
    // The API and implementation of this function is based on
    // https://pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html

    c10::OptionalDeviceGuard device_guard{capture_stream_.device()};

    // Just like any RNG consumer kernel!
    auto* gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
    at::PhiloxCudaState rng_engine_inputs;
    {
      std::lock_guard<std::mutex> lock(gen->mutex_);
      rng_engine_inputs = gen->philox_cuda_state(wholegraph_increment_);
    }
    seed_extragraph_.fill_(int64_t(gen->current_seed()));
    offset_extragraph_.fill_(int64_t(rng_engine_inputs.offset_.val));
  }

  c10::cuda::MempoolId_t pool() {
    // The API and implementation of this function is based on
    // https://pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html
    return mempool_id_;
  }

 private:
  // uuid of this instance's current capture, used to
  // specify the pool.
  c10::cuda::CaptureId_t id_;

  // the ID assigned by cuda during graph capture,
  // used to identify when a stream is participating in capture
  c10::cuda::CaptureId_t capture_id_ = -1;

  // uuid used to request a particular private mempool from
  // CUDACachingAllocator. By default, this will be set to {id_, 0}.
  //
  // If capture_begin is called with "pool=other_graph.pool()", this graph's
  // mempool_id_ will be set to the other graph's mempool_id_, and therefore
  // share a mempool with the other graph.
  //
  // If capture_begin is called with "pool=handle" where "handle" came from
  // graph_pool_handle(), it will share a mempool with any other captures that
  // used "pool=handle".
  //
  // Sharing a mempool across graphs saves memory, and it's safe if you
  // know you'll replay those graphs in the same order you captured them.
  c10::cuda::MempoolId_t mempool_id_;

  // Stream on which capture began
  at::cuda::CUDAStream capture_stream_;

  // Default generator on device where capture began
  at::CUDAGeneratorImpl* capture_gen_;

  // Device where capture occurred. Right now, for simplicity, we require all
  // ops in a capture to run on the same device, but this is a limitation of
  // CUDAGraph, not CUDA itself.  We can straightforwardly modify CUDAGraph to
  // support multi-device captures if needed.
  int capture_dev_;

  // RNG state trackers
  at::Tensor seed_extragraph_;
  at::Tensor offset_extragraph_;
  uint64_t wholegraph_increment_;
};

// No need to inherit from torch::CustomClassHolder because the holder type is
// std::shared_ptr instead of c10::intrusive_ptr
class PyWrapperCudaGraphWrapper {
 public:
  PyWrapperCudaGraphWrapper(cudaGraph_t graph)
      : graph_wrapper_(std::make_shared<CudaGraphWrapper>(graph)) {}
  PyWrapperCudaGraphWrapper(long graph)
      : graph_wrapper_(std::make_shared<CudaGraphWrapper>(
            static_cast<cudaGraph_t>(reinterpret_cast<void*>(graph)))) {}
  cudaGraph_t getGraph() { return graph_wrapper_->get<cudaGraph_t>(); }
  void notifyAddedAsChildGraph() { graph_wrapper_->notifyAddedAsChildGraph(); }
  bool isAddedAsChildGraph() { return graph_wrapper_->addedAsChildGraph; }
  void executeGraph(cudaStream_t stream) {
    graph_wrapper_->executeGraph(stream);
  }
  void destroyGraphExec() { graph_wrapper_->destroyGraphExec(); }
  void addGraphAsChildNode(PyWrapperCudaGraphWrapper& child_graph) {
    graph_wrapper_->addGraphAsChildNode((*child_graph.getGraphWrapper()));
  }
  std::shared_ptr<CudaGraphWrapper> getGraphWrapper() { return graph_wrapper_; }

 private:
  std::shared_ptr<CudaGraphWrapper> graph_wrapper_;
};

/// CUDA Graph Constructors Class methods
// No need to inherit from torch::CustomClassHolder because the holder type is
// std::shared_ptr instead of c10::intrusive_ptr
class PyWrapperCUDAGraphConstructor {
 public:
  PyWrapperCUDAGraphConstructor() : constructor_() {}
  void registerStream(long stream) {
    void* stream_v = reinterpret_cast<void*>(stream);
    cudaStream_t stream_t = static_cast<cudaStream_t>(stream_v);
    constructor_.registerStream(stream_t);
  }

  void addEventRecordNode(long event, long stream) {
    void* event_v = reinterpret_cast<void*>(event);
    cudaEvent_t event_t = static_cast<cudaEvent_t>(event_v);
    void* stream_v = reinterpret_cast<void*>(stream);
    cudaStream_t stream_t = static_cast<cudaStream_t>(stream_v);
    constructor_.addEventRecordNode(event_t, stream_t);
  }

  void addStreamWaitEventNode(long stream, long event) {
    void* stream_v = reinterpret_cast<void*>(stream);
    cudaStream_t stream_t = static_cast<cudaStream_t>(stream_v);
    void* event_v = reinterpret_cast<void*>(event);
    cudaEvent_t event_t = static_cast<cudaEvent_t>(event_v);
    constructor_.addStreamWaitEventNode(stream_t, event_t);
  }

  void join(std::vector<long> streams, long dst_stream) {
    std::vector<cudaStream_t> streams_t;
    for (auto stream : streams) {
      void* stream_v = reinterpret_cast<void*>(stream);
      cudaStream_t stream_t = static_cast<cudaStream_t>(stream_v);
      streams_t.push_back(stream_t);
    }
    void* dst_stream_v = reinterpret_cast<void*>(dst_stream);
    cudaStream_t dst_stream_t = static_cast<cudaStream_t>(dst_stream_v);
    constructor_.join(streams_t, dst_stream_t);
  }

  void executeGraph(cudaStream_t stream) {
    constructor_.getGraphWrapper()->executeGraph(stream);
  }

  void destroyGraphExec() {
    constructor_.getGraphWrapper()->destroyGraphExec();
  }

  void notifyBeforeInvokingLibraryCall(long stream) {
    void* stream_v = reinterpret_cast<void*>(stream);
    cudaStream_t stream_t = static_cast<cudaStream_t>(stream_v);
    constructor_.notifyBeforeInvokingLibraryCall(stream_t);
  }

  void notifyAfterInvokingLibraryCall(long stream) {
    void* stream_v = reinterpret_cast<void*>(stream);
    cudaStream_t stream_t = static_cast<cudaStream_t>(stream_v);
    constructor_.notifyAfterInvokingLibraryCall(stream_t);
  }

  void addGraphAsChildNode(PyWrapperCudaGraphWrapper& child_graph) {
    constructor_.getGraphWrapper()->addGraphAsChildNode(
        (*child_graph.getGraphWrapper()));
  }

  void addGraphAsChildNode(PyWrapperCUDAGraphConstructor& child_graph) {
    constructor_.getGraphWrapper()->addGraphAsChildNode(
        (*child_graph.constructor_.getGraphWrapper()));
  }

  template <typename T>
  T combineGraphs(PyWrapperCUDAGraphConstructor& graph_constructor_rhs) {
    LOG(WARNING)
        << " The scheme is not working in some cases because it add the two "
           "graphs as child graphs, which does not support the GEMM graph "
           "that "
           "contains memset node when the kernel is large.";
    // Merge the two graphs and then launch it
    cudaGraph_t merged_graph = combineCUDAGraphs(std::vector<cudaGraph_t>{
        constructor_.getGraph(),
        graph_constructor_rhs.constructor_.getGraph()});
    // Avoid double freeing the child graph, i.e., the graph of the GEMM
    // partitioned since the destruction of parent graph will destroy it
    // already
    constructor_.getGraphWrapper()->notifyAddedAsChildGraph();
    graph_constructor_rhs.constructor_.getGraphWrapper()
        ->notifyAddedAsChildGraph();
    if constexpr (std::is_same<T, long>::value) {
      return PyLong_AsLong(PyLong_FromVoidPtr(merged_graph));
    } else if constexpr (std::is_same<T,
                                      std::shared_ptr<
                                          PyWrapperCudaGraphWrapper>>::value) {
      return std::make_shared<PyWrapperCudaGraphWrapper>(merged_graph);
    } else {
      throw std::runtime_error("Unsupported return type");
    }
  }

  void dumpGraph(const std::string& result_path) {
    constructor_.dumpGraph(result_path);
  }

 private:
  CUDAExperimentalGraphConstructor<cudaStream_t> constructor_;
};
}  // namespace
}  // namespace IntraSMEngine

// From
// https://github.com/pytorch/pytorch/blob/ff4aac109a990e64d82fb73b5af5fa5e69278580/torch/csrc/cuda/Graph.cpp
// unless specified, py::class_ uses std::unique_ptr<T> as the holder type.
// Reference:
// https://pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html#:~:text=the%20default%20for%20a%20type%20named
template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  /// Exporting methods
  m.def("print_cudastream", &IntraSMEngine::print_cudastream,
        "print cudaStream_t");
  m.def("print_cudastream", &IntraSMEngine::print_cudastreams,
        "print multiple cudaStream_t (overloaded API)");
  m.def("print_cudaevent", &IntraSMEngine::print_cudaevent,
        "print cudaEvent_t");

  /// Exporting class definitions
  shared_ptr_class_<IntraSMEngine::CUDAGraphCaptureNotifier>(
      m, "CUDAGraphCaptureNotifier")
      .def(py::init<>())
      .def(
          "capture_begin",
          [](IntraSMEngine::CUDAGraphCaptureNotifier& self,
             c10::optional<c10::cuda::MempoolId_t> pool_opt) {
            c10::cuda::MempoolId_t pool = pool_opt.has_value()
                                              ? pool_opt.value()
                                              : c10::cuda::MempoolId_t{0, 0};
            return self.capture_begin(pool);
          },
          // Reference:
          // https://pybind11.readthedocs.io/en/stable/advanced/functions.html#non-converting-arguments:~:text=A%20py%3A%3Aargs%20argument%20implies%20that
          py::arg("pool"),
          // Reference:
          // https://pybind11.readthedocs.io/en/stable/advanced/functions.html#call-guard
          py::call_guard<py::gil_scoped_release>())
      .def("capture_end",
           torch::wrap_pybind_function_no_gil(
               &IntraSMEngine::CUDAGraphCaptureNotifier::capture_end))
      .def("replay", torch::wrap_pybind_function_no_gil(
                         &IntraSMEngine::CUDAGraphCaptureNotifier::replay))
      .def("pool", torch::wrap_pybind_function_no_gil(
                       &IntraSMEngine::CUDAGraphCaptureNotifier::pool))
      .def("assert_capture_has_begun",
           torch::wrap_pybind_function_no_gil(
               &IntraSMEngine::CUDAGraphCaptureNotifier::
                   assert_capture_has_begun));

  shared_ptr_class_<IntraSMEngine::PyWrapperCudaGraphWrapper>(
      m, "CudaGraphWrapper")
      .def(py::init<long>())
      .def("notify_added_as_child_graph",
           &IntraSMEngine::PyWrapperCudaGraphWrapper::notifyAddedAsChildGraph)
      .def("add_graph_as_child_node",
           torch::wrap_pybind_function_no_gil(
               &IntraSMEngine::PyWrapperCudaGraphWrapper::addGraphAsChildNode))
      .def(
          "execute_graph",
          [](IntraSMEngine::PyWrapperCudaGraphWrapper& self, long stream) {
            void* stream_v = reinterpret_cast<void*>(stream);
            cudaStream_t stream_t = static_cast<cudaStream_t>(stream_v);
            self.executeGraph(stream_t);
          },
          py::arg("stream"), py::call_guard<py::gil_scoped_release>())
      .def(
          "destroy_graph_exec",
          torch::wrap_pybind_function_no_gil(
              &IntraSMEngine::PyWrapperCUDAGraphConstructor::destroyGraphExec));

  shared_ptr_class_<IntraSMEngine::PyWrapperCUDAGraphConstructor>(
      m, "CUDAExperimentalGraphConstructor")
      .def(py::init<>())
      .def(
          "execute_graph",
          [](IntraSMEngine::PyWrapperCUDAGraphConstructor& self, long stream) {
            void* stream_v = reinterpret_cast<void*>(stream);
            cudaStream_t stream_t = static_cast<cudaStream_t>(stream_v);
            self.executeGraph(stream_t);
          },
          py::arg("stream"), py::call_guard<py::gil_scoped_release>())
      .def("destroy_graph_exec",
           torch::wrap_pybind_function_no_gil(
               &IntraSMEngine::PyWrapperCUDAGraphConstructor::destroyGraphExec))
      .def("register_stream",
           torch::wrap_pybind_function_no_gil(
               &IntraSMEngine::PyWrapperCUDAGraphConstructor::registerStream))
      .def("add_event_record_node",
           torch::wrap_pybind_function_no_gil(
               &IntraSMEngine::PyWrapperCUDAGraphConstructor::
                   addEventRecordNode))
      .def("add_stream_wait_event_node",
           torch::wrap_pybind_function_no_gil(
               &IntraSMEngine::PyWrapperCUDAGraphConstructor::
                   addStreamWaitEventNode))
      .def("join", torch::wrap_pybind_function_no_gil(
                       &IntraSMEngine::PyWrapperCUDAGraphConstructor::join))
      .def("notify_before_invoking_library_call",
           torch::wrap_pybind_function_no_gil(
               &IntraSMEngine::PyWrapperCUDAGraphConstructor::
                   notifyBeforeInvokingLibraryCall))
      .def("notify_after_invoking_library_call",
           torch::wrap_pybind_function_no_gil(
               &IntraSMEngine::PyWrapperCUDAGraphConstructor::
                   notifyAfterInvokingLibraryCall))
      .def("add_graph_as_child_node",
           torch::wrap_pybind_function_no_gil(
               static_cast<void (
                   IntraSMEngine::PyWrapperCUDAGraphConstructor::*)(
                   IntraSMEngine::PyWrapperCudaGraphWrapper&)>(
                   &IntraSMEngine::PyWrapperCUDAGraphConstructor::
                       addGraphAsChildNode)))
      .def("add_graph_as_child_node",
           torch::wrap_pybind_function_no_gil(
               static_cast<void (
                   IntraSMEngine::PyWrapperCUDAGraphConstructor::*)(
                   IntraSMEngine::PyWrapperCUDAGraphConstructor&)>(
                   &IntraSMEngine::PyWrapperCUDAGraphConstructor::
                       addGraphAsChildNode)))
      .def("combine_graphs",
           torch::wrap_pybind_function_no_gil(
               &IntraSMEngine::PyWrapperCUDAGraphConstructor::combineGraphs<
                   long>))
      .def("dump_graph",
           torch::wrap_pybind_function_no_gil(
               &IntraSMEngine::PyWrapperCUDAGraphConstructor::dumpGraph));
}
