// TODO: move to the intrasm_engine repo. Makefile changes are in the
// dev_ie_migration branch.
#pragma once
#include <assert.h>
#include <cuda_runtime.h>

#include <vector>

#include "helper_cuda_errors.cu.h"

// This function instantiates the graph before execution. It could also be used
// to update the cudaGraphExec_t in some scenarios, e.g., when the node
// parameters are updated. Notice that if the graph is unchanged, it can be
// executed multiple times without re-instantiation/update. See
// https://github.com/NVIDIA/cuda-samples/blob/master/Samples/4_CUDA_Libraries/conjugateGradientCudaGraphs/conjugateGradientCudaGraphs.cu
// for an example.
void cudaGraphInitiateOrExecUpdate(cudaGraph_t graph,
                                   cudaGraphExec_t *pGraphExec) {
  if (*pGraphExec == NULL) {
    checkCudaErrors(cudaGraphInstantiate(pGraphExec, graph, NULL, NULL, 0));
  } else {
    cudaGraphExecUpdateResultInfo updateResult_out;
    checkCudaErrors(cudaGraphExecUpdate(*pGraphExec, graph, &updateResult_out));
    if (updateResult_out.result != cudaGraphExecUpdateSuccess) {
      if (*pGraphExec != NULL) {
        checkCudaErrors(cudaGraphExecDestroy(*pGraphExec));
      }
      printf("graph update failed with error - %d\n", updateResult_out.result);
      checkCudaErrors(cudaGraphInstantiate(pGraphExec, graph, NULL, NULL, 0));
    }
  }
}

cudaGraph_t combineCUDAGraphs(std::vector<cudaGraph_t> graphs) {
  // Create the combined graph following example in
  // python/notebooks/experiment_with_cuda_python/[WORKING]manipulate_cublas_graph.py
  // The returned graph needs to be initiated before execution.
  // The scheme is not working in some cases because it add the two graphs as
  // child graphs, which does not support the GEMM graph that contains memset
  // node when the kernel is large.
  cudaGraph_t combinedGraph;
  checkCudaErrors(cudaGraphCreate(&combinedGraph, 0));
  cudaGraphNode_t root;
  cudaGraphNode_t firstNode;
  // It seems the returned cudaGraphNode_t firstNode, a pointer, is NULL. Maybe
  // it is a bug in CUDA software.
  checkCudaErrors(cudaGraphAddEmptyNode(&firstNode, combinedGraph, NULL, 0));
  size_t numRootNodes;
  checkCudaErrors(cudaGraphGetRootNodes(combinedGraph, &root, &numRootNodes));

  // printf("numRootNodes %d\n", numRootNodes);
  // printf("root %x\n", root);
  assert(numRootNodes == 1 && "numRootNodes should be 1");
  for (auto graph : graphs) {
    cudaGraphNode_t newNode;
    checkCudaErrors(cudaGraphAddChildGraphNode(
        /*pGraphNode*/ &newNode, /*graph*/ combinedGraph,
        /*pDependencies*/ &root, /*numDependencies*/ 1, /*childGraph*/ graph));
  }

  return combinedGraph;
}

void updateKernelNodeInGraph(void *func, dim3 nblocks, dim3 nthreads,
                             cudaGraphExec_t graphExec,
                             cudaGraphNode_t kernelNode, void **kernelArgs) {
  cudaKernelNodeParams NodeParams;
  NodeParams.func = func;
  NodeParams.gridDim = nblocks;
  NodeParams.blockDim = nthreads;
  NodeParams.sharedMemBytes = 0;
  NodeParams.kernelParams = kernelArgs;
  NodeParams.extra = NULL;

  checkCudaErrors(
      cudaGraphExecKernelNodeSetParams(graphExec, kernelNode, &NodeParams));
}

cudaGraphNode_t createKernelNodeInGraph(
    void *func, dim3 nblocks, dim3 nthreads, cudaGraph_t graph,
    std::vector<cudaGraphNode_t> nodeDependencies, void **kernelArgs) {
  cudaGraphNode_t kernelNode;
  cudaKernelNodeParams NodeParams;
  NodeParams.func = func;
  NodeParams.gridDim = nblocks;
  NodeParams.blockDim = nthreads;
  NodeParams.sharedMemBytes = 0;
  // void *kernelArgs0[6] = {(void *)&A, (void *)&b,     (void
  // *)&conv_threshold,
  //                         (void *)&x, (void *)&x_new, (void *)&d_sum};
  NodeParams.kernelParams = kernelArgs;
  NodeParams.extra = NULL;

  checkCudaErrors(cudaGraphAddKernelNode(&kernelNode, graph,
                                         nodeDependencies.data(),
                                         nodeDependencies.size(), &NodeParams));
  return kernelNode;
}

cudaGraphNode_t createHostNodeInGraph(
    cudaHostFn_t
        func,  // The func must have the signature void(*)(void* userData)
    cudaGraph_t graph, std::vector<cudaGraphNode_t> nodeDependencies,
    void *hostArgs) {
  cudaGraphNode_t hostNode;
  cudaHostNodeParams NodeParams;
  NodeParams.fn = func;
  NodeParams.userData = hostArgs;
  checkCudaErrors(cudaGraphAddHostNode(&hostNode, graph,
                                       nodeDependencies.data(),
                                       nodeDependencies.size(), &NodeParams));
  return hostNode;
}

// Use CUDA graph via the stream capture method. From
// JacobiMethodGpuCudaGraphExecUpdate() from
// https://github.com/NVIDIA/cuda-samples/blob/master/Samples/3_CUDA_Features/jacobiCudaGraphs/jacobi.cu
int cuda_graph_example_skeleton(int max_iter = 100) {
  cudaStream_t stream1;
  checkCudaErrors(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));
  cudaGraph_t graph;
  cudaGraphExec_t graphExec = NULL;
  // Alternative method: Step 1/4 checkCudaErrors(cudaGraphCreate(&graph, 0));
  // Alternative method: Step 2/4 add kernel and/or memcpy node to graph
  // cudaGraphInitiateOrExecUpdate(graph, &graphExec);

  // Alternative method: Step 3/4 cudGraphInstantiate needs only one call and is
  // before the loop
  for (int k = 0; k < max_iter; k++) {
    checkCudaErrors(
        cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal));

    // DO: Kernel and MemcpyAsync using stream1

    checkCudaErrors(cudaStreamEndCapture(stream1, &graph));

    cudaGraphInitiateOrExecUpdate(graph, &graphExec);

    // Alternative method: Step 4/4 If only node parameter is changed but the
    // graph is the same across iteration, set kernel param to node.
    checkCudaErrors(cudaGraphLaunch(graphExec, stream1));
    checkCudaErrors(cudaStreamSynchronize(stream1));
  }
  checkCudaErrors(cudaStreamDestroy(stream1));
  return 0;
}

// TODO: cuStreamUpdateCaptureDependencies seems like an option to remove the
// artificial dependency in the captured stream
// Another option is dependencies_out in cuStreamGetCaptureInfo
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g378135b262f02a43a7caeab239ae493d