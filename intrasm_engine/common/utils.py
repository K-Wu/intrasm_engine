import git
import hashlib
import subprocess


def get_repo_hash():
    """Return the concatenation of commit hash and hash of `git diff`"""
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    diff = repo.git.diff()
    # print(sha, hashlib.sha1(diff.encode()).hexdigest())
    # Concatenate the two hashes together
    return sha + hashlib.sha1(diff.encode()).hexdigest()


# From /HET/hrt/utils/stat_sass_inst.py
def demangle_cuda_function_name(func_name: str) -> str:
    """Demangle CUDA function name."""
    return (
        subprocess.check_output(["cu++filt", func_name])
        .decode("utf-8")
        .strip()
    )


if __name__ == "__main__":
    import os

    # Read the symbol table .txt at the same directory as this script
    with open(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "sputnik_cuda_spmm_symbol_table.txt",
        )
    ) as fd:
        for line in fd:
            line = line.strip()
            if line.startswith("STT_FUNC"):
                mangled_func_name = line.split(" ")[-1]
                print(
                    mangled_func_name,
                    demangle_cuda_function_name(mangled_func_name),
                )
