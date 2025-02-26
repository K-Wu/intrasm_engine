// Based on
// https://github.com/facebookresearch/xformers/blob/1191f035e8aa06241850dedf2c20d66f5b3ba575/xformers/csrc/attention/cuda/spmm.cu
// The file is to provided SpMM kernel modified from xformers+sputnik to 1)
// implement batch support and 2) switch the store operation from store to
// atomicAdd for // This file was modified from sputnik to implement batch
// support for spmm.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>

#include <cmath>
#include <functional>
#include <string>
#include <unordered_map>

#include "sputnik/barrier.h"
#include "sputnik/cuda_utils.h"
#include "sputnik/load_store.h"
#include "sputnik/memory_aligner.h"
#include "sputnik/spmm/compute_utils.h"
#include "sputnik/spmm/cuda_spmm.h"
#include "sputnik/spmm/dense_tile.h"
#include "sputnik/spmm/output_tile.h"
#ifdef ENABLE_ATOMIC_STORE_FLAG
#include "sputnik/spmm/output_tile_atomic.h"
#endif
#include "sputnik/spmm/predicate_utils.h"
#include "sputnik/spmm/sparse_tile.h"
#include "sputnik/spmm/spmm_config.h"
#include "sputnik/tiling_utils.h"
#include "sputnik/vector_utils.h"

#define CONCAT_ID_(prefix, id) prefix##id
#define CONCAT_ID(prefix, id) CONCAT_ID_(prefix, id)

namespace my_sputnik {
using namespace sputnik;

template <typename Config>
cudaError_t CONCAT_ID(CudaSpmmEx, MYID)(
    int m, int k, int n, int nonzeros, const int* __restrict__ row_indices,
    const typename Config::ScalarValue* __restrict__ values,
    const int* __restrict__ row_offsets,
    const typename Config::ScalarIndex* __restrict__ column_indices,
    const typename Config::ScalarValue* __restrict__ dense_matrix,
    const float* __restrict__ bias,
    typename Config::ScalarValue* __restrict__ output_matrix,
    cudaStream_t stream, int batch_size);

namespace {

template <typename Config>
struct SpmmKernel2 {
  //
  /// Shortcuts for commonly used specialized types.
  //

  typedef TilingUtils<Config::kBlockItemsY, Config::kBlockItemsK,
                      Config::kBlockItemsX>
      Tiling;

  typedef PredicateVector<Config::kThreadItemsX> PredicateVector;

  typedef PredicatesN<typename Config::DenseValue, Config::kBlockItemsX,
                      Config::kBlockWidth>
      PredicatesN;

  typedef MemoryAligner<typename Config::SparseValue, Config::kBlockWidth>
      MemoryAligner;

  typedef SparseTile<typename Config::SparseValue, Config::kBlockItemsK,
                     Config::kBlockWidth>
      SparseTile;

  typedef DenseTile<typename Config::DenseValue, Config::kBlockItemsK,
                    Config::kBlockItemsX, Config::kBlockWidth,
                    Config::kResidueUnroll>
      DenseTile;

  typedef ComputeUtils<typename Config::DenseValue, Config::kBlockItemsK,
                       Config::kBlockItemsX, Config::kBlockWidth>
      Computer;

  typedef Barrier<Config::kBlockItemsY, Config::kBlockWidth> Barrier;

  typedef OutputTile<typename Config::DenseValue, Config::kBlockItemsX,
                     Config::kBlockWidth>
      OutputTile;

  typedef typename Config::ScalarValue ScalarValue;
  typedef typename Config::DenseValue DenseValue;
  typedef typename Config::SparseValue SparseValue;
  typedef typename Config::ScalarIndex ScalarIndex;
  typedef typename Config::Index Index;

  /**
   * @brief Main function for SpMM kernel.
   */
  static __device__ __forceinline__ void KernelFn(
      int m, int k, int n, const int* __restrict__ row_indices,
      const ScalarValue* __restrict__ values,
      const int* __restrict__ row_offsets,
      const ScalarIndex* __restrict__ column_indices,
      const ScalarValue* __restrict__ dense_matrix,
      const float* __restrict__ bias, ScalarValue* __restrict__ out, int nnz) {
    // Calculate this thread block's indices into the M and N dimensions.
    int m_index = Tiling::IndexM(), n_index = Tiling::IndexN();

    // Threads that work on different m-dim indices are independent. If
    // we're out of bounds in the m-dimension we can just return.
    if (m_index >= m) return;
    m_index = Load(row_indices + m_index);

    // Divide some of our constant problem dimensions and indices by
    // the number of elements that are packed into each scalar.
    n /= Config::kElementsPerScalar;

    // Initialize the n-dimension predicates for this thread.
    PredicateVector predicates_n;
    if (Config::kPredicateLoads) {
      PredicatesN::Set(n_index, n, &predicates_n);
    }

    // Load the row offset and calculate the number of non-zeros in the row.
    int row_offset = Load(row_offsets + m_index);
    int nonzeros = Load(row_offsets + m_index + 1) - row_offset;

    // Divide some of our constant values by the number of elements that
    // are packed into a single scalar.
    nonzeros /= Config::kElementsPerScalar;
    row_offset /= Config::kElementsPerScalar;

    // Possibly align the row offset s.t. it's safe to use vector memory ops.
    //
    // Note that if we only have residue to process, we do not align the row
    // offset. This lets us not worry about masking in the residue handling,
    // where we use scalar memory operations anyways.
    MemoryAligner memory_aligner(row_offset, nonzeros);
    int aligned_nonzeros = memory_aligner.AlignedNonzeros();
    if (aligned_nonzeros >= Config::kBlockItemsK) {
      nonzeros = aligned_nonzeros;
      row_offset = memory_aligner.AlignedRowOffset();
    }

    // Shared memory tiles for the lhs values and indices.
    constexpr int kTileSize = Config::kBlockItemsK * Config::kBlockItemsY;
    __shared__ ScalarValue values_tile_array[kTileSize];
    __shared__ ScalarIndex column_indices_tile_array[kTileSize];

    // Possibly increment our tile pointers for 2D tiling schemes.
    ScalarValue* values_tile = Tiling::MaybeOffset(
        values_tile_array, Config::kBlockItemsK * threadIdx.y);
    ScalarIndex* column_indices_tile = Tiling::MaybeOffset(
        column_indices_tile_array, Config::kBlockItemsK * threadIdx.y);

    // Create a loader for the sparse lhs matrix.
    SparseTile sparse_tile_loader(n, row_offset, threadIdx.x,
#ifdef ENABLE_REUSE_WEIGHT_FLAG
                                  values,
#else
                                  values + blockIdx.z * nnz,
#endif
                                  column_indices, values_tile,
                                  column_indices_tile);

    // Register fragment for the dense_matrix values.
    constexpr int kDenseFragmentSize =
        Config::kElementsPerScalar * Config::kBlockItemsK *
        Config::kBlockItemsX / Config::kBlockWidth;
    __align__(16) ScalarValue dense_matrix_fragment[kDenseFragmentSize];

    // Create a loader for the dense dense_matrix matrix.
    DenseTile dense_tile_loader(n, n_index, threadIdx.x,
#ifdef ENABLE_REUSE_RHS_FLAG
                                dense_matrix,
#else
                                dense_matrix + blockIdx.z * k * n,
#endif
                                column_indices_tile, dense_matrix_fragment);

    // Accumulator registers for the output values. Initialize the
    // registers to zero s.t. we can always accumulate in-place.
    constexpr int kOutputFragmentSize =
        Config::kBlockItemsX / Config::kBlockWidth * Config::kElementsPerScalar;
    __align__(16) float output_fragment[kOutputFragmentSize] = {};

    // Helper for computing tile-level partial matmuls.
    Computer computer(values_tile, dense_matrix_fragment, output_fragment);

    // Helper for managing synchronization between collaborating threads.
    Barrier barrier(threadIdx.y);

    //
    /// Begin kernel main loop.
    //

    // For the first iteration of our main loop, we need to possibly mask
    // the first few values from the sparse matrix in case we aligned our
    // values and column indices pointers.
    if (nonzeros >= Config::kBlockItemsK) {
      // Load a tile from the sparse lhs matrix and synchronize the cta.
      sparse_tile_loader.Load();
      barrier.Sync();

      // Mask any values we loaded that aren't from our row of the sparse
      // matrix. Threads could potentially mask values in smem that they
      // were not responsible for loading. Synchronize again to make sure
      // the masking occurs after the previous loads have completed.
      //
      // TODO(tgale): We don't need to synchronize here for the scalar
      // variants of the kernels. We also don't need to handle the first
      // iteration specially. This kernel has now become very complex. It
      // would be nice to break it out into an SpMM class where we can
      // break each of these sections out into helper functions.
      memory_aligner.MaskPrefix(values_tile, column_indices_tile);
      barrier.Sync();

      // Load a tile from the sparse dense_matrix matrix.
      dense_tile_loader.Load(predicates_n);

      // Multiply the tiles and accumulate the results.
      computer.TileMAC();
      nonzeros -= Config::kBlockItemsK;
    }

    // Loop over the tiles in the k-dimension of the dense_matrix/lhs matrices.
    for (; nonzeros >= Config::kBlockItemsK; nonzeros -= Config::kBlockItemsK) {
      // Synchronize s.t. we don't overwrite our shared memory tiles while
      // other warps have not completed using them in computation.
      barrier.Sync();

      // Load a tile from the sparse lhs matrix and synchronize the cta.
      sparse_tile_loader.Load();
      barrier.Sync();

      // Load a tile from the sparse dense_matrix matrix.
      dense_tile_loader.Load(predicates_n);

      // Multiply the tiles and accumulate the results.
      computer.TileMAC();
    }

    //
    /// Begin spmm residue computation.
    //

    // Synchronize s.t. we don't overwrite our shared memory tiles while
    // other warps have not completed using them in computation.
    barrier.Sync();

    // Zero the shared memory tiles s.t. we can operate on sets of 2/4
    // values safely in the dense tile loads and computation.
    if (Config::kResidueUnroll > 1) {
      sparse_tile_loader.ZeroTiles();
      barrier.Sync();
    }

    // Load a tile from the sparse lhs matrix and synchronize the cta.
    sparse_tile_loader.Residue(nonzeros);
    barrier.Sync();

    // Load a tile from the dense dense_matrix matrix and compute immediately.
    dense_tile_loader.ResidueLoadAndCompute(nonzeros, predicates_n, values_tile,
                                            output_fragment);

    //
    /// Write results to the output.
    //

    // Possibly apply the bias and RelU.
    if (bias != nullptr) {
      // Bias value is shared across all outputs.
      const float bias_value = Load(bias + m_index);
#pragma unroll
      for (int out_idx = 0; out_idx < kOutputFragmentSize; ++out_idx) {
        output_fragment[out_idx] += bias_value;
        output_fragment[out_idx] =
            output_fragment[out_idx] > 0 ? output_fragment[out_idx] : 0;
      }
    }

    // Create a storer for the output matrix.
    OutputTile output_tile_storer(m_index, n_index, n, threadIdx.x,
                                  output_fragment, out + blockIdx.z * m * n);
#ifdef ENABLE_ATOMIC_STORE_FLAG
    sputnik::MyStore<typename Config::DenseValue, Config::kBlockItemsX,
                     Config::kBlockWidth, true>(output_tile_storer,
                                                predicates_n);
#else
    output_tile_storer.Store(predicates_n);
#endif
  }
};

template <typename Config>
__global__ void __launch_bounds__(Config::kThreadsPerBlock)
    CONCAT_ID(Kernel, MYID)(
        int m, int k, int n, const int* __restrict__ row_indices,
        const typename Config::ScalarValue* __restrict__ values,
        const int* __restrict__ row_offsets,
        const typename Config::ScalarIndex* __restrict__ column_indices,
        const typename Config::ScalarValue* __restrict__ dense_matrix,
        const float* __restrict__ bias,
        typename Config::ScalarValue* __restrict__ out, int nnz) {
  SpmmKernel2<Config>::KernelFn(m, k, n, row_indices, values, row_offsets,
                                column_indices, dense_matrix, bias, out, nnz);
}

template <typename Config>
__global__ void __launch_bounds__(Config::kThreadsPerBlock,
                                  Config::kMinOccupancy)
    CONCAT_ID(KernelWithBounds, MYID)(
        int m, int k, int n, const int* __restrict__ row_indices,
        const typename Config::ScalarValue* __restrict__ values,
        const int* __restrict__ row_offsets,
        const typename Config::ScalarIndex* __restrict__ column_indices,
        const typename Config::ScalarValue* __restrict__ dense_matrix,
        const float* __restrict__ bias,
        typename Config::ScalarValue* __restrict__ out, int nnz) {
  SpmmKernel2<Config>::KernelFn(m, k, n, row_indices, values, row_offsets,
                                column_indices, dense_matrix, bias, out, nnz);
}

std::string MakeHandle(int m, int k, int n, int nonzeros) {
  // NOTE: We don't include the number of nonzeros currently.
  return std::to_string(m) + "_" + std::to_string(k) + "_" + std::to_string(n);
}

}  // namespace

template <typename Config>
cudaError_t CONCAT_ID(CudaSpmmEx, MYID)(
    int m, int k, int n, int nonzeros, const int* __restrict__ row_indices,
    const typename Config::ScalarValue* __restrict__ values,
    const int* __restrict__ row_offsets,
    const typename Config::ScalarIndex* __restrict__ column_indices,
    const typename Config::ScalarValue* __restrict__ dense_matrix,
    const float* __restrict__ bias,
    typename Config::ScalarValue* __restrict__ output_matrix,
    cudaStream_t stream, int batch_size) {
  dim3 grid_dim(ceil(static_cast<float>(m) / Config::kBlockItemsY),
                ceil(static_cast<float>(n) / Config::kBlockItemsX /
                     Config::kElementsPerScalar),
                batch_size);
  dim3 block_dim(Config::kBlockWidth, Config::kBlockItemsY, 1);

  if (Config::kLaunchBounds) {
    CONCAT_ID(KernelWithBounds, MYID)<Config>
        <<<grid_dim, block_dim, 0, stream>>>(
            m, k, n, row_indices, values, row_offsets, column_indices,
            dense_matrix, bias, output_matrix, nonzeros);
  } else {
    CONCAT_ID(Kernel, MYID)<Config><<<grid_dim, block_dim, 0, stream>>>(
        m, k, n, row_indices, values, row_offsets, column_indices, dense_matrix,
        bias, output_matrix, nonzeros);
  }
  return cudaGetLastError();
}
}  // namespace my_sputnik
