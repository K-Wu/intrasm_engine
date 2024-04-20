// TODO: move to the intrasm_engine repo. Makefile changes are in the
// dev_ie_migration branch.
#pragma once
#include <cuda_runtime.h>

// TODO: Follow the optimization from Version 5, which is the best, at
// github.com/NVIDIA/cuda-samples/Samples/2_Concepts_and_Techniques/reduction~/cuda-samples/Samples/2_Concepts_and_Techniques/reduction
template <int BLOCK_SIZE, int SHMEM_SIZE, typename DataType>
__global__ void reduce_segments(DataType *d_in, DataType *d_out, int ldxx,
                                int dimxx, int dimyy, int ldx, int dimx,
                                int seg_count) {
  // ldxx is the leading dimension and y is the other dimension.
  // For each accumulated output element, a total of BLOCK_SIZE/SHMEM_SIZE
  // threads contribute to it. Finally, SHMEM_SIZE threads are used to write
  // the output.

  // The outermost loop level is the k/kk dimension, and then the y dimension,
  // and the innermost is the x dimension.

  // launch configuration: 2D grid of 1D blocks. Each block has BLOCK_SIZE.
  // gridDim.x == x*y/SHMEM_SIZE. gridDim.y determines seg_count each block
  // handles.

  // If the intent is instead to sum up each segment, i.e., in another direction
  // to what this function offers, check thrust::reduce_by_key example at
  // https://github.com/NVIDIA/thrust/blob/master/examples/sum_rows.cu

  __shared__ float sdata[SHMEM_SIZE];
  if (threadIdx.x < SHMEM_SIZE) {
    sdata[threadIdx.x] = 0.;
  }
  __syncthreads();
  int ele_id = blockIdx.x * SHMEM_SIZE + threadIdx.x % SHMEM_SIZE;
  int curr_x = ele_id % dimx;
  int curr_y = ele_id / dimx;
  int curr_xx = ele_id % dimxx;
  int curr_yy = ele_id / dimxx;

  int dimy = seg_count * dimyy;

  for (int seg = seg_count / (gridDim.y * BLOCK_SIZE / SHMEM_SIZE) *
                     blockIdx.y * (BLOCK_SIZE / SHMEM_SIZE) +
                 threadIdx.x / SHMEM_SIZE;
       seg < seg_count / (gridDim.y * BLOCK_SIZE / SHMEM_SIZE) *
                     (blockIdx.y + 1) * (BLOCK_SIZE / SHMEM_SIZE) +
                 threadIdx.x / SHMEM_SIZE;
       seg += BLOCK_SIZE / SHMEM_SIZE) {
    if (curr_xx < dimxx && curr_yy < dimyy) {
      atomicAdd(&sdata[threadIdx.x % SHMEM_SIZE],
                d_in[seg_count * ldx * dimy + curr_xx + curr_yy * ldxx]);
    }
  }

  __syncthreads();
  if (threadIdx.x < SHMEM_SIZE) {
    d_out[curr_x + curr_y * ldx] = sdata[threadIdx.x];
  }
}
