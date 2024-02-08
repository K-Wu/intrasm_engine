import random

"""
sparse matrices provided by sgk dump is in the format of smtx, and it stores
only positions without values. However, the file loader in mlir only supports
.mtx. This scripts randomly genereates values for the smtx files and convert
them to mtx files.
"""


def generate_mtx_from_smtx(smtx_filename):
    # mtx file format:
    # %%MatrixMarket matrix coordinate real general
    # num_rows num_cols num_nonzeros
    # row col value on every line without comma

    # smtx file format:
    # num_rows, num_cols, num_nonzeros
    # row_ptrs without comma
    # col_idxs without comma

    # read smtx file
    with open(smtx_filename, "r") as f:
        lines = f.readlines()
        num_rows, num_cols, num_nonzeros = lines[0].strip().split(",")
        num_rows = int(num_rows)
        num_cols = int(num_cols)
        num_nonzeros = int(num_nonzeros)
        row_ptrs = lines[1].strip().split(" ")
        col_idxs = lines[2].strip().split(" ")
        row_ptrs = [int(i) for i in row_ptrs if i != ""]  # Skip empty lines
        col_idxs = [int(i) for i in col_idxs if i != ""]

    # generate values
    values = [random.random() for i in range(num_nonzeros)]
    with open(smtx_filename[: smtx_filename.rfind(".")] + ".mtx", "w") as fd:
        fd.write("%%MatrixMarket matrix coordinate real general\n")
        fd.write("{} {} {}\n".format(num_rows, num_cols, num_nonzeros))
        for i in range(num_rows):
            for j in range(row_ptrs[i], row_ptrs[i + 1]):
                fd.write(
                    "{} {} {}\n".format(i + 1, col_idxs[j] + 1, values[j])
                )


if __name__ == "__main__":
    # find all .smtx in data/, and convert them to .mtx
    import os

    dlmc_data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        "data",
        "sgk_dlmc",
    )
    print("Working on DLMC data path:", dlmc_data_path)
    for root, dirs, files in os.walk(dlmc_data_path):
        print("Now in", root, dirs, files)
        for file in files:
            if file.endswith(".smtx"):
                # Skip if the mtx file already exists
                if os.path.exists(
                    os.path.join(root, file[: file.rfind(".")] + ".mtx")
                ):
                    continue
                print("Converting", os.path.join(root, file))
                generate_mtx_from_smtx(os.path.join(root, file))
