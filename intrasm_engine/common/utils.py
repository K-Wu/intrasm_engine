import git
import hashlib
import os


# From https://stackoverflow.com/a/41920796/5555077
def get_git_root(path: str) -> str:
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root


def get_git_root_recursively(path: str) -> str:
    """Get the git root path by recursively searching parent directories until the repo root is no longer a submodule."""
    current_path = path
    while True:
        git_repo = git.Repo(current_path, search_parent_directories=True)
        git_root = get_git_root(current_path)
        if (
            len(git_repo.git.rev_parse("--show-superproject-working-tree"))
            == 0
        ):
            return os.path.normpath(git_root)
        else:
            current_path = os.path.dirname(git_root)


def get_repo_hash() -> str:
    """Return the concatenation of commit hash and hash of `git diff`"""
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    diff = repo.git.diff()
    # print(sha, hashlib.sha1(diff.encode()).hexdigest())
    # Concatenate the two hashes together
    return sha + hashlib.sha1(diff.encode()).hexdigest()


if __name__ == "__main__":
    print(get_git_root_recursively(os.path.dirname(__file__)))
