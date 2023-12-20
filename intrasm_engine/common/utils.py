import git
import hashlib


def get_repo_hash():
    """Return the concatenation of commit hash and hash of `git diff`"""
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    diff = repo.git.diff()
    # print(sha, hashlib.sha1(diff.encode()).hexdigest())
    # Concatenate the two hashes together
    return sha + hashlib.sha1(diff.encode()).hexdigest()
