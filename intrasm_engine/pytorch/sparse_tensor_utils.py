import torch
from typing import Any
from xformers import sparse


def partition_csr(
    crow_indices: torch.Tensor,
    col_indices: torch.Tensor,
    values: torch.Tensor,
    num_rows: int,
    num_cols: int,
    num_rows_per_partition: int,
    num_cols_per_partition: int,
) -> tuple[
    list[list[torch.Tensor]],
    list[list[torch.Tensor]],
    list[list[torch.Tensor]],
]:
    """
    Partition a CSR matrix into a grid of submatrices.
    """

    crow_indices_list = [
        [
            torch.zeros(num_rows_per_partition + 1, dtype=torch.int64)
            for _2 in range(num_cols // num_cols_per_partition)
        ]
        for _ in range(num_rows // num_rows_per_partition)
    ]

    # [[[]] * (num_cols // num_cols_per_partition)] * (num_rows // num_rows_per_partition)
    col_indices_list: list[list[Any]] = [
        [[]] * (num_cols // num_cols_per_partition)
    ] * (num_rows // num_rows_per_partition)
    values_list: list[list[Any]] = [
        [[]] * (num_cols // num_cols_per_partition)
    ] * (num_rows // num_rows_per_partition)
    for idx_row_partition in range(num_rows // num_rows_per_partition):
        for idx_row_element in range(
            num_rows // num_rows_per_partition * idx_row_partition,
            num_rows // num_rows_per_partition * (idx_row_partition + 1),
        ):
            for element_idx in range(
                crow_indices[idx_row_element],
                crow_indices[idx_row_element + 1],
            ):
                idx_col_partition = (
                    col_indices[element_idx] // num_cols_per_partition
                )
                crow_indices_list[idx_row_partition][idx_col_partition][
                    idx_row_element % num_rows_per_partition
                ] += 1
                col_indices_list[idx_row_partition][idx_col_partition].append(
                    col_indices[element_idx] % num_cols_per_partition
                )
                values_list[idx_row_partition][idx_col_partition].append(
                    values[element_idx]
                )
    for idx_row_partition in range(num_rows // num_rows_per_partition):
        for idx_col_partition in range(num_cols // num_cols_per_partition):
            # Accumulate crow_indices in crow_indices_list
            crow_indices_list[idx_row_partition][
                idx_col_partition
            ] = crow_indices_list[idx_row_partition][idx_col_partition].cumsum(
                0
            )

            # Convert list to torch.Tensor
            col_indices_list[idx_row_partition][
                idx_col_partition
            ] = torch.tensor(
                col_indices_list[idx_row_partition][idx_col_partition],
                dtype=torch.int64,
            )
            values_list[idx_row_partition][idx_col_partition] = torch.tensor(
                values_list[idx_row_partition][idx_col_partition],
                dtype=torch.float32,
            )
    return crow_indices_list, col_indices_list, values_list


def partition_coo(
    row_indices: torch.Tensor,
    col_indices: torch.Tensor,
    values: torch.Tensor,
    num_rows: int,
    num_cols: int,
    num_rows_per_partition: int,
    num_cols_per_partition: int,
) -> tuple[
    list[list[torch.Tensor]],
    list[list[torch.Tensor]],
    list[list[torch.Tensor]],
]:
    """
    Partition a COO matrix into a grid of submatrices.
    """

    row_indices_list: list[list[Any]] = [
        [[]] * (num_cols // num_cols_per_partition)
    ] * (num_rows // num_rows_per_partition)
    col_indices_list: list[list[Any]] = [
        [[]] * (num_cols // num_cols_per_partition)
    ] * (num_rows // num_rows_per_partition)
    values_list: list[list[Any]] = [
        [[]] * (num_cols // num_cols_per_partition)
    ] * (num_rows // num_rows_per_partition)
    for element_idx in range(row_indices.shape[0]):
        idx_col_partition = col_indices[element_idx] // num_cols_per_partition
        idx_row_partition = row_indices[element_idx] // num_rows_per_partition
        row_indices_list[idx_row_partition][idx_col_partition].append(
            row_indices[element_idx] % num_rows_per_partition
        )
        col_indices_list[idx_row_partition][idx_col_partition].append(
            col_indices[element_idx] % num_cols_per_partition
        )
        values_list[idx_row_partition][idx_col_partition].append(
            values[element_idx]
        )
    for idx_row_partition in range(num_rows // num_rows_per_partition):
        for idx_col_partition in range(num_cols // num_cols_per_partition):
            # Convert list to torch.Tensor
            row_indices_list[idx_row_partition][
                idx_col_partition
            ] = torch.tensor(
                row_indices_list[idx_row_partition][idx_col_partition],
                dtype=torch.int64,
            )
            col_indices_list[idx_row_partition][
                idx_col_partition
            ] = torch.tensor(
                col_indices_list[idx_row_partition][idx_col_partition],
                dtype=torch.int64,
            )
            values_list[idx_row_partition][idx_col_partition] = torch.tensor(
                values_list[idx_row_partition][idx_col_partition],
                dtype=torch.float32,
            )
    return row_indices_list, col_indices_list, values_list


# Conversion functions from dense tensor with zeros to sparse tensor. Use xformers/xformers/sparse/utils.py
def dense_to_sparse(
    matrix, device=torch.device(f"cuda:{torch.cuda.current_device()}")
):
    """Converts dense 2d matrix to a csr sparse matrix."""
    sparse.utils._dense_to_sparse(matrix, device)
