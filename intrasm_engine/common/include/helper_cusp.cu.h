// TODO: move to the intrasm_engine repo. Makefile changes are in the
// dev_ie_migration branch.
#pragma once
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>

#include <iostream>
#include <vector>

template <typename DstMatrixType, typename SrcMatrixType>
std::vector<DstMatrixType> convertCuspMatrices(
    std::vector<SrcMatrixType> &src_matrices) {
  std::vector<DstMatrixType> dst_matrices;
  for (auto &src_matrix : src_matrices) {
    DstMatrixType dst_matrix(src_matrix);
    dst_matrices.push_back(dst_matrix);
  }
  return dst_matrices;
}

template <typename IdxType, typename ValueType, typename MemorySpace>
bool isEqual(cusp::coo_matrix<IdxType, ValueType, MemorySpace> &A,
             cusp::coo_matrix<IdxType, ValueType, MemorySpace> &B) {
  if (A.num_entries != B.num_entries) {
    printf("A.num_entries != B.num_entries\n");
    return false;
  }
  if (A.num_rows != B.num_rows) {
    printf("A.num_rows != B.num_rows\n");
    return false;
  }
  if (A.num_cols != B.num_cols) {
    printf("A.num_cols != B.num_cols\n");
    return false;
  }
  cusp::coo_matrix<IdxType, ValueType, MemorySpace> A_sorted(A);
  cusp::coo_matrix<IdxType, ValueType, MemorySpace> B_sorted(B);
  A_sorted.sort_by_row_and_column();
  B_sorted.sort_by_row_and_column();

  bool is_equal = true;
  for (int i = 0; i < A_sorted.num_entries; i++) {
    if (A_sorted.row_indices[i] != B_sorted.row_indices[i]) {
      printf("A_sorted.row_indices[%d] != B_sorted.row_indices[%d]\n", i, i);
      std::cout << A_sorted.row_indices[i] << " " << B_sorted.row_indices[i]
                << std::endl;
      is_equal = false;
    }
    if (A_sorted.column_indices[i] != B_sorted.column_indices[i]) {
      printf("A_sorted.column_indices[%d] != B_sorted.column_indices[%d]\n", i,
             i);
      std::cout << A_sorted.column_indices[i] << " "
                << B_sorted.column_indices[i] << std::endl;
      is_equal = false;
    }
    if (A_sorted.values[i] != B_sorted.values[i]) {
      printf("A_sorted.values[%d] != B_sorted.values[%d]\n", i, i);
      std::cout << A_sorted.values[i] << " " << B_sorted.values[i] << std::endl;
      is_equal = false;
    }
  }
  return is_equal;
}

cusp::coo_matrix<int, float, cusp::host_memory> combineMatrices(
    std::vector<cusp::coo_matrix<int, float, cusp::host_memory>> &matrices,
    int num_partitions_along_row, int num_partitions_along_col) {
  int num_rows = 0;
  int num_cols = 0;
  int num_entries = 0;
  assert(matrices.size() ==
         num_partitions_along_row * num_partitions_along_col);

  for (int idx_partition_col = 0; idx_partition_col < num_partitions_along_col;
       idx_partition_col++) {
    int idx_partition =
        /* idx_row = */ 0 + idx_partition_col * num_partitions_along_row;
    num_cols += matrices[idx_partition].num_cols;
  }

  for (int idx_partition_row = 0; idx_partition_row < num_partitions_along_row;
       idx_partition_row++) {
    int idx_partition =
        idx_partition_row + /* idx_col = */ 0 * num_partitions_along_row;
    num_rows += matrices[idx_partition].num_rows;
  }

  int num_rows_per_partition = num_rows / num_partitions_along_row;
  int num_cols_per_partition = num_cols / num_partitions_along_col;

  for (int idx_partition_col = 0; idx_partition_col < num_partitions_along_col;
       idx_partition_col++) {
    for (int idx_partition_row = 0;
         idx_partition_row < num_partitions_along_row; idx_partition_row++) {
      int idx_partition =
          idx_partition_row + idx_partition_col * num_partitions_along_row;
      auto &matrix = matrices[idx_partition];
      num_entries += matrix.num_entries;
    }
  }
  cusp::coo_matrix<int, float, cusp::host_memory> combined(num_rows, num_cols,
                                                           num_entries);

  int entry_offset = 0;
  for (int idx_partition_col = 0; idx_partition_col < num_partitions_along_col;
       idx_partition_col++) {
    for (int idx_partition_row = 0;
         idx_partition_row < num_partitions_along_row; idx_partition_row++) {
      int idx_partition =
          idx_partition_row + idx_partition_col * num_partitions_along_row;
      auto &matrix = matrices[idx_partition];
      for (int i = 0; i < matrix.num_entries; i++) {
        combined.row_indices[entry_offset + i] =
            matrix.row_indices[i] + idx_partition_row * num_rows_per_partition;
        combined.column_indices[entry_offset + i] =
            matrix.column_indices[i] +
            idx_partition_col * num_cols_per_partition;
        combined.values[entry_offset + i] = matrix.values[i];
      }
      entry_offset += matrix.num_entries;
    }
  }

  return combined;
}

std::vector<cusp::coo_matrix<int, float, cusp::host_memory>> partitionMatrix(
    cusp::coo_matrix<int, float, cusp::host_memory> &A,
    int num_partitions_along_row, int num_partitions_along_col) {
  std::vector<cusp::coo_matrix<int, float, cusp::host_memory>> partitions;
  int num_partitions = num_partitions_along_row * num_partitions_along_col;
  int num_rows_per_partition = A.num_rows / num_partitions_along_row;
  int num_cols_per_partition = A.num_cols / num_partitions_along_col;
  std::vector<cusp::array1d<int, cusp::host_memory>> row_indices(
      num_partitions);
  std::vector<cusp::array1d<int, cusp::host_memory>> col_indices(
      num_partitions);
  std::vector<cusp::array1d<float, cusp::host_memory>> values(num_partitions);
  for (int i = 0; i < A.num_entries; i++) {
    int row = A.row_indices[i];
    int col = A.column_indices[i];
    int partition_rowIdx = row / num_rows_per_partition;
    int partition_colIdx = col / num_cols_per_partition;
    // Column major
    int partition_id =
        partition_rowIdx + partition_colIdx * num_partitions_along_row;

    // Need to recalculate row and col for each partition
    row_indices[partition_id].push_back(row % num_rows_per_partition);
    col_indices[partition_id].push_back(col % num_cols_per_partition);
    values[partition_id].push_back(A.values[i]);
  }
  for (int i = 0; i < num_partitions; i++) {
    cusp::coo_matrix<int, float, cusp::host_memory> partition(
        num_rows_per_partition, num_cols_per_partition, row_indices[i].size());
    std::copy(row_indices[i].begin(), row_indices[i].end(),
              partition.row_indices.begin());
    std::copy(col_indices[i].begin(), col_indices[i].end(),
              partition.column_indices.begin());
    std::copy(values[i].begin(), values[i].end(), partition.values.begin());

    partitions.push_back(partition);
  }
  return partitions;
}

std::vector<cusp::csr_matrix<int, float, cusp::host_memory>> partitionMatrix(
    cusp::csr_matrix<int, float, cusp::host_memory> &A,
    int num_partitions_along_row, int num_partitions_along_col) {
  cusp::coo_matrix<int, float, cusp::host_memory> A_coo(A);
  auto partitions_coo = partitionMatrix(A_coo, num_partitions_along_row,
                                        num_partitions_along_col);
  auto partitions_csr =
      convertCuspMatrices<cusp::csr_matrix<int, float, cusp::host_memory>,
                          cusp::coo_matrix<int, float, cusp::host_memory>>(
          partitions_coo);
  return partitions_csr;
}