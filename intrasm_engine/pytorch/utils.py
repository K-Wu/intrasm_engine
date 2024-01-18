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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Converts dense 2d matrix to a csr sparse matrix."""
    (
        values,
        row_indices,
        row_offsets,
        column_indices,
    ) = sparse.utils._dense_to_sparse(matrix, device)
    return values, row_indices, row_offsets, column_indices


# TODO: extract mask from pytorch model parameters, e.g., https://huggingface.co/SparseLLM
# Mask reference: random_mask in intrasm_engine/3rdparty/SparTA/sparta/testing/mask.py


def check_tensor_eligible_for_tensor_core(a: torch.Tensor):
    # Tensor cores are used if the inputs are float16, the shapes are multiples of 8 and it’s a matmul call.
    # From https://discuss.pytorch.org/t/how-to-check-tensor-core-has-been-used/39714
    for i in a.shape:
        if i % 8 != 0:
            return False
    if a.dtype != torch.float16:
        return False
    if a.is_cuda:
        return True
    return False


def set_tf32_use_tensor_core():
    print("Setting TF32 use Tensor Core")
    # From https://discuss.pytorch.org/t/does-pytorch-use-tensor-cores-by-default/167676/3
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True
    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True


def reset_tf32_use_tensor_core():
    print("Resetting TF32 use Tensor Core")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def set_float_16_reduced_precision():
    print("Setting float16 and bf16 using reduced precision in reduction")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True


def reset_float_16_reduced_precision():
    print("Resetting float16 and bf16 using reduced precision in reduction")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
