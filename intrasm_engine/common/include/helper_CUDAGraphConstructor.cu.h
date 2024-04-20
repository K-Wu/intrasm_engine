// TODO: move to the intrasm_engine repo. Makefile changes are in the
// dev_ie_migration branch.
#pragma once
#include <cuda_runtime.h>

#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "assert.h"

#define ENABLE_DEBUG_PRINTx
#ifdef ENABLE_DEBUG_PRINT
#define debug_printf(...) printf(__VA_ARGS__)
#else
#define debug_printf(...)
#endif

#ifndef CUDA_CHECK
// CUDA API error checking
#define CUDA_CHECK(err)                                                  \
  do {                                                                   \
    cudaError_t err_ = (err);                                            \
    if (err_ != cudaSuccess) {                                           \
      std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("CUDA error");                            \
    }                                                                    \
  } while (0)
#endif

class CudaGraphAdjacencyMonitor {
 private:
  cudaGraph_t graph;

 public:
  std::map<cudaGraphNode_t, std::set<cudaGraphNode_t>> adjacencyMap;
  CudaGraphAdjacencyMonitor(cudaGraph_t graph) : graph(graph) {
    notifyAddNodes();
  }
  void _addEdge(cudaGraphNode_t from, cudaGraphNode_t to) {
    adjacencyMap[from].insert(to);
  }
  void notifyAddNodes() {
    size_t numNodes;
    std::set<cudaGraphNode_t> newlyAddedNodes;
    // Get number of edges first.
    // Figure out newly-added nodes
    cudaGraphGetNodes(graph, nullptr, &numNodes);
    std::vector<cudaGraphNode_t> nodes(numNodes);
    cudaGraphGetNodes(graph, nodes.data(), &numNodes);
    for (size_t i = 0; i < numNodes; i++) {
      if (adjacencyMap.find(nodes[i]) == adjacencyMap.end()) {
        // The node is newly added to the graph
        adjacencyMap[nodes[i]] = std::set<cudaGraphNode_t>();
        newlyAddedNodes.insert(nodes[i]);
      }
    }

    size_t numEdges = 0;
    cudaGraphGetEdges(graph, nullptr, nullptr, &numEdges);
    std::vector<cudaGraphNode_t> edgesFrom(numEdges);
    std::vector<cudaGraphNode_t> edgesTo(numEdges);
    cudaGraphGetEdges(graph, edgesFrom.data(), edgesTo.data(), &numEdges);
    for (size_t i = 0; i < numEdges; i++) {
      // Update the adjacency map of newly-added nodes
      if (newlyAddedNodes.find(edgesFrom[i]) != newlyAddedNodes.end()) {
        adjacencyMap[edgesFrom[i]].insert(edgesTo[i]);
        continue;
      }
      // Update the adjacency map of existing nodes to added nodes
      if (newlyAddedNodes.find(edgesTo[i]) != newlyAddedNodes.end()) {
        std::set<cudaGraphNode_t>& currNodeTos = adjacencyMap[edgesFrom[i]];
        if (currNodeTos.find(edgesTo[i]) != currNodeTos.end()) {
          throw std::runtime_error("Duplicate edge");
        }
        currNodeTos.insert(edgesTo[i]);
      }
    }
  }

  cudaGraphNode_t goToEndOfChain(cudaGraphNode_t node) {
    // This function aims at finding the end of a chain of nodes. It is used to
    // update the last activity on a StreamOrToken after capturing activities
    // into the existing graph. Check notifyAfterInvokingLibraryCall() in
    // CUDAGraphConstructor for details.
    while (adjacencyMap[node].size() == 1) {
      node = *adjacencyMap[node].begin();
    }
    assert(adjacencyMap[node].size() == 0 && "Node should be the end of chain");
    return node;
  }

  void print() {
    for (auto& [from, tos] : adjacencyMap) {
      std::cout << "Node " << from << " has edges to ";
      for (auto& to : tos) {
        std::cout << to << " ";
      }
      std::cout << std::endl;
    }
  }
};

struct CudaGraphWrapper {
  cudaGraph_t graph = nullptr;
  cudaGraphNode_t root = nullptr;
  cudaGraphExec_t graphExec = nullptr;
  bool addedAsChildGraph =
      false;  // This flag indicates whether the graph is added as a child
              // node of another graph. If so, we cannot destroy the graph in
              // destructor because it causes double-free.
  bool disableDestructionOtherReason = false;
  CudaGraphWrapper() {
    CUDA_CHECK(cudaGraphCreate(&graph, 0));
    CUDA_CHECK(cudaGraphAddEmptyNode(&root, graph, nullptr, 0));
  }
  void instantiateGraphExec() {
    if (graphExec != nullptr) {
      throw std::runtime_error(
          "The graph has already been executed. Please wait for the end of "
          "the "
          "graph execution finish and reset graphExec by destroyGraphExec() "
          "before running it again.");
    }
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
  }
  void executeGraph(cudaStream_t stream) { cudaGraphLaunch(graphExec, stream); }
  void destroyGraphExec() {
    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    graphExec = nullptr;
  }
  void addGraphAsChildNode(CudaGraphWrapper& child_graph) {
    cudaGraphNode_t child_graph_node;
    CUDA_CHECK(cudaGraphAddChildGraphNode(&child_graph_node, graph, &root, 1,
                                          child_graph.get<cudaGraph_t>()));
    // Avoid double freeing the child graph
    child_graph.notifyAddedAsChildGraph();
  }
  CudaGraphWrapper(const cudaGraph_t graph) {
    this->graph = graph;
    size_t num_roots = 0;
    CUDA_CHECK(cudaGraphGetRootNodes(graph, &root, &num_roots));
    assert(num_roots == 1 && "numRootNodes should be 1");
  }
  ~CudaGraphWrapper() {
    if (graphExec != nullptr) {
      destroyGraphExec();
    }
    if (!addedAsChildGraph && !disableDestructionOtherReason)
      CUDA_CHECK(cudaGraphDestroy(graph));
  }
  template <typename T>
  T get() {
    if constexpr (std::is_same<T, cudaGraph_t>::value) {
      return graph;
    } else if constexpr (std::is_same<T, cudaGraphNode_t>::value) {
      return root;
    } else {
      throw std::runtime_error("Invalid type");
    }
  }
  void notifyAddedAsChildGraph() { addedAsChildGraph = true; }
  void notifyExternalManagement() { disableDestructionOtherReason = true; }
};

struct cudaStreamWrapper {
  cudaStream_t stream = nullptr;
  cudaStreamWrapper() {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
  ~cudaStreamWrapper() { CUDA_CHECK(cudaStreamDestroy(stream)); }
  cudaStream_t get() { return stream; }
};

struct cudaEventWrapper {
  cudaEvent_t event;
  cudaEventWrapper() { CUDA_CHECK(cudaEventCreate(&event)); }
  ~cudaEventWrapper() { CUDA_CHECK(cudaEventDestroy(event)); }
  cudaEvent_t get() { return event; }
};

template <typename StreamOrToken>
class AbstractCUDAGraphConstructor {
 public:
  virtual void print() = 0;
  virtual void registerStream(StreamOrToken stream) = 0;
  virtual void registerStreamLastActivity(StreamOrToken stream,
                                          cudaGraphNode_t node) = 0;
  virtual void addEventRecordNode(cudaEvent_t event, StreamOrToken stream) = 0;
  virtual void addStreamWaitEventNode(StreamOrToken stream,
                                      cudaEvent_t event) = 0;
  virtual void join(std::vector<StreamOrToken> streams,
                    StreamOrToken dst_stream) = 0;
  // The two functions update the stream_last_activity map, and either 1) do
  // stream capture to create child graph node in CUDAGraphConstructor, or 2)
  // do nothing in CUDAStreamConstructor.
  virtual void notifyBeforeInvokingLibraryCall(StreamOrToken stream) = 0;
  virtual void notifyAfterInvokingLibraryCall(StreamOrToken stream) = 0;
};

class CUDAStreamConstructor
    : public AbstractCUDAGraphConstructor<cudaStream_t> {
 private:
  std::map<cudaStream_t, cudaEvent_t>
      streamLastActivityMap;  // Used when joining streams. If the last
                              // activity is an event, then the event
                              // is used. In other cases, the value should be
                              // nullptr, and we need to create a new event.
  std::vector<std::shared_ptr<cudaEventWrapper>>
      temp_events_for_joins;  // Temporary events created in the join()
                              // execution for synchronization, to be
                              // destroyed in the destructor.

 public:
  CUDAStreamConstructor() {}

  void print() override { std::cout << "CUDAStreamConstructor" << std::endl; }

  void registerStreamLastActivity(cudaStream_t stream, cudaGraphNode_t node) {
    throw std::runtime_error(
        "CUDAStreamConstructor did not implement registerStreamLastActivity "
        "but use streamLastActivityMap to store events");
  }

  void registerStream(cudaStream_t stream) override {
    streamLastActivityMap[stream] = nullptr;
  }

  void addEventRecordNode(cudaEvent_t event, cudaStream_t stream) override {
    CUDA_CHECK(cudaEventRecord(event, stream));
    streamLastActivityMap[stream] = event;
  }

  void addStreamWaitEventNode(cudaStream_t stream, cudaEvent_t event) override {
    CUDA_CHECK(cudaStreamWaitEvent(stream, event, 0));
  }

  void join(std::vector<cudaStream_t> streams,
            cudaStream_t dst_stream) override {
    for (auto stream : streams) {
      cudaEvent_t event = streamLastActivityMap[stream];
      if (event == nullptr) {
        temp_events_for_joins.push_back(
            std::make_shared<cudaEventWrapper>());  // Create a new event
        event = temp_events_for_joins.back()->get();
        CUDA_CHECK(cudaEventRecord(event, stream));
        streamLastActivityMap[stream] =
            event;  // Update the map in case there is another join operation
      }
      CUDA_CHECK(cudaStreamWaitEvent(dst_stream, event, 0));
      streamLastActivityMap[dst_stream] = nullptr;
    }
  }

  void notifyBeforeInvokingLibraryCall(cudaStream_t stream) override {
    return;  // Do nothing
  }

  void notifyAfterInvokingLibraryCall(cudaStream_t stream) override {
    registerStreamLastActivity(stream,
                               nullptr);  // Reset the map, meaning the
    // last activity is no longer an event
  }
};

template <typename StreamOrToken>
class CUDAGraphConstructor
    : public AbstractCUDAGraphConstructor<StreamOrToken> {
 protected:
  std::shared_ptr<CudaGraphWrapper> graph;
  std::shared_ptr<cudaStreamWrapper>
      defaultStream;  // Default stream when StreamOrToken is token.
  std::map<StreamOrToken, cudaGraphNode_t>
      streamLastActivityMap;  // Used when joining streams. If the last
                              // activity is an event, then the event
                              // is the node; if the last activity is a
                              // library call, then the node is a subgraph.
                              // In both cases, we add the node as a
                              // dependency.

 public:
  CUDAGraphConstructor() {
#if defined(USE_ROCM)
    throw std::runtime_error("CUDAGraphConstructor is not supported on ROCM");
#endif
    graph = std::make_shared<CudaGraphWrapper>();
    defaultStream = std::make_shared<cudaStreamWrapper>();
  }

  void print() override { std::cout << "CUDAGraphConstructor" << std::endl; }
  cudaGraph_t getGraph() { return graph->get<cudaGraph_t>(); }

  void dumpGraph(const std::string& result_path) {
    unsigned int dumpFlags = cudaGraphDebugDotFlagsVerbose |
                             cudaGraphDebugDotFlagsKernelNodeParams |
                             cudaGraphDebugDotFlagsMemcpyNodeParams |
                             cudaGraphDebugDotFlagsMemsetNodeParams |
                             cudaGraphDebugDotFlagsHostNodeParams |
                             cudaGraphDebugDotFlagsEventNodeParams |
                             cudaGraphDebugDotFlagsExtSemasSignalNodeParams |
                             cudaGraphDebugDotFlagsExtSemasWaitNodeParams |
                             cudaGraphDebugDotFlagsKernelNodeAttributes |
                             cudaGraphDebugDotFlagsHandles;
    CUDA_CHECK(cudaGraphDebugDotPrint(graph->get<cudaGraph_t>(),
                                      result_path.c_str(), dumpFlags));
  }

  void registerStreamLastActivity(StreamOrToken stream, cudaGraphNode_t node) {
    streamLastActivityMap[stream] = node;
  }

  void registerStream(StreamOrToken stream) override {
    debug_printf("registerStream\n");
    debug_printf("stream %p\n", stream);
    debug_printf("streamLastActivityMap[stream] %p\n",
                 graph->get<cudaGraphNode_t>());
    streamLastActivityMap[stream] = graph->get<cudaGraphNode_t>();
  }

  void addEventRecordNode(cudaEvent_t event, StreamOrToken stream) override {
    cudaGraphNode_t event_node;
    cudaGraphNode_t last_node = streamLastActivityMap[stream];
    CUDA_CHECK(cudaGraphAddEventRecordNode(
        &event_node, graph->get<cudaGraph_t>(), &last_node, 1, event));
    registerStreamLastActivity(stream, event_node);
  }

  void addStreamWaitEventNode(StreamOrToken stream,
                              cudaEvent_t event) override {
    cudaGraphNode_t wait_node;
    CUDA_CHECK(cudaGraphAddEventWaitNode(&wait_node, graph->get<cudaGraph_t>(),
                                         &streamLastActivityMap[stream], 1,
                                         event));
    registerStreamLastActivity(stream, wait_node);
  }

  void join(std::vector<StreamOrToken> streams,
            StreamOrToken dst_stream) override {
    cudaGraphNode_t join_node;
    std::vector<cudaGraphNode_t> dependencies;
    for (auto stream : streams) {
      dependencies.push_back(streamLastActivityMap[stream]);
    }
    CUDA_CHECK(cudaGraphAddEmptyNode(&join_node, graph->get<cudaGraph_t>(),
                                     dependencies.data(), dependencies.size()));
    registerStreamLastActivity(dst_stream, join_node);
  }

  virtual void _notifyBeforeInvokingLibraryCall(cudaStream_t stream) {
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  }

  virtual void _notifyAfterInvokingLibraryCall(cudaStream_t stream) {
    cudaGraph_t child_graph;
    CUDA_CHECK(cudaStreamEndCapture(defaultStream->get(), &child_graph));
    cudaGraphNode_t child_graph_node;
    CUDA_CHECK(cudaGraphAddChildGraphNode(
        &child_graph_node, graph->get<cudaGraph_t>(),
        &streamLastActivityMap[stream], 1, child_graph));
    registerStreamLastActivity(stream, child_graph_node);
  }

  /// The following are logic of functions when StreamOrToken is token;
  /// cudaStream_t version is implemented via partial specialization at the
  /// end of this header file. The difference is that when StreamOrToken is
  /// token, we will use a default stream because the argument is a token.
  /// Otherwise we will use the stream specified in the arguments.

  // NB: capturing and add as subgraph won't work with routines that
  // involve memset, e.g., large cuBLAS GEMM calls.
  // Please use CUDAExperimentalGraphConstructor instead if you have such
  // need.
  void notifyBeforeInvokingLibraryCall(StreamOrToken stream) override {
    _notifyBeforeInvokingLibraryCall(defaultStream->get());
  }

  // NB: capturing and add as subgraph won't work with routines that
  // involve memset, e.g., large cuBLAS GEMM calls.
  // Please use CUDAExperimentalGraphConstructor instead if you have such
  // need.
  void notifyAfterInvokingLibraryCall(StreamOrToken stream) override {
    _notifyAfterInvokingLibraryCall(defaultStream->get());
  }

  std::shared_ptr<CudaGraphWrapper> getGraphWrapper() { return graph; };
};

/// The following are logic of functions when StreamOrToken is stream.
/// The difference is that when StreamOrToken is token, we will use a default
/// stream because the argument is a token. Otherwise we will use the stream
/// specified in the arguments.
template <>
void CUDAGraphConstructor<cudaStream_t>::notifyBeforeInvokingLibraryCall(
    cudaStream_t stream) {
  _notifyBeforeInvokingLibraryCall(stream);
}

template <>
void CUDAGraphConstructor<cudaStream_t>::notifyAfterInvokingLibraryCall(
    cudaStream_t stream) {
  _notifyAfterInvokingLibraryCall(stream);
}

template <typename StreamOrToken>
class CUDAExperimentalGraphConstructor
    : public CUDAGraphConstructor<StreamOrToken> {
 private:
  // std::set<cudaGraphNode_t> nodesHasBeenLastActivityOfOneStream;
  CudaGraphAdjacencyMonitor adjacencyMonitor;

 public:
  CUDAExperimentalGraphConstructor()
      : CUDAGraphConstructor<StreamOrToken>(),
        adjacencyMonitor(CUDAGraphConstructor<StreamOrToken>::getGraphWrapper()
                             ->template get<cudaGraph_t>()) {}

  void registerStreamLastActivity(StreamOrToken stream, cudaGraphNode_t node) {
    CUDAGraphConstructor<StreamOrToken>::streamLastActivityMap[stream] = node;
    // nodesHasBeenLastActivityOfOneStream.insert(node);
  }

  CudaGraphAdjacencyMonitor& getAdjacencyMonitor() { return adjacencyMonitor; }

  // Capture and add to graph. Only works in CUDA 12.3+
  void _notifyBeforeInvokingLibraryCall(cudaStream_t stream) override {
    debug_printf("notifyBeforeInvokingLibraryCall\n");

    debug_printf(
        "streamLastActivityMap[stream] %p\n",
        CUDAGraphConstructor<StreamOrToken>::streamLastActivityMap[stream]);
    debug_printf("stream %p\n", stream);
    debug_printf("graph %p\n", (CUDAGraphConstructor<StreamOrToken>::graph)
                                   ->template get<cudaGraph_t>());
#if ((__CUDACC_VER_MAJOR__ > 12) || \
     ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ >= 3)))
    // CUDA >= 12.3
    CUDA_CHECK(cudaStreamBeginCaptureToGraph(
        stream,
        (CUDAGraphConstructor<StreamOrToken>::graph)
            ->template get<cudaGraph_t>(),
        &(CUDAGraphConstructor<StreamOrToken>::streamLastActivityMap)[stream],
        nullptr, 1, cudaStreamCaptureModeGlobal));
#else
    throw std::runtime_error(
        "notifyBeforeInvokingLibraryCall() only works in CUDA 12.3+");
#endif
  }

  void _notifyAfterInvokingLibraryCall(cudaStream_t stream) override {
#if ((__CUDACC_VER_MAJOR__ > 12) || \
     ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ >= 3)))
    cudaGraph_t graph_;
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph_));
    // Find out the graph node and update the streamLastActivityMap record
    size_t numEdges = 0;
    // Get number of edges first.
    cudaGraphGetEdges(graph_, nullptr, nullptr, &numEdges);
    std::vector<cudaGraphNode_t> edgesFrom(numEdges);
    std::vector<cudaGraphNode_t> edgesTo(numEdges);
    cudaGraphGetEdges(graph_, edgesFrom.data(), edgesTo.data(), &numEdges);
    for (size_t i = 0; i < numEdges; i++) {
      if (edgesFrom[i] == (CUDAGraphConstructor<
                              StreamOrToken>::streamLastActivityMap)[stream]) {
        if (adjacencyMonitor.adjacencyMap.find(edgesTo[i]) !=
            adjacencyMonitor.adjacencyMap.end()) {
          continue;
        }
        debug_printf("Found the graph node\n");
        adjacencyMonitor.notifyAddNodes();
        debug_printf("After notifyAddNodes\n");
        // Walk to the end of edgesTo[i]
        auto node = adjacencyMonitor.goToEndOfChain(edgesTo[i]);
        debug_printf("After goToEndOfChain\n");
        registerStreamLastActivity(stream, node);
        return;
      }
    }
    throw std::runtime_error("Cannot find the graph node");
#else
    throw std::runtime_error(
        "notifyAfterInvokingLibraryCall() only works in CUDA 12.3+");
#endif
  }
};