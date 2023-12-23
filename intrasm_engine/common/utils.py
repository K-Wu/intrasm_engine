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
    print(
        demangle_cuda_function_name(
            "_ZN3cub18DeviceReduceKernelINS_18Device"
            "ReducePolicyIN6thrust5tupleIblNS2_9null"
            "_typeES4_S4_S4_S4_S4_S4_S4_EElNS2_8cuda"
            "_cub9__find_if7functorIS5"
            "_EEE9Policy600ENS2_12zip_iteratorINS3"
            "_INS6_26transform_input_iterator_tIbNS6"
            "_35transform_pair_of_input_iterators"
            "_tIbNS2_6detail15normal_iteratorINS2"
            "_10device_ptrIKfEEEESK_NS2_8equal"
            "_toIfEEEENSF_12unary_negateINS6"
            "_8identityEEEEENS6_19counting_iterator"
            "_tIlEES4_S4_S4_S4_S4_S4_S4_S4_EEEElS9"
            "_S5_EEvT0_PT3_T1_NS_13GridEvenShareISZ"
            "_EET2_"
        )
    )
