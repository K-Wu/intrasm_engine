def reverse_chthill_mckee_example():
    # From https://stackoverflow.com/questions/75532847/reverse-cuthill-mckee-permutation-on-sparse-coo-matrix
    import numpy as np
    from scipy import sparse
    from scipy.sparse import csr_matrix, coo_matrix

    # Documentation is at https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.reverse_cuthill_mckee.html
    from scipy.sparse.csgraph import reverse_cuthill_mckee
    import matplotlib.pyplot as plt

    npt = 100
    fmat = sparse.rand(npt, npt, density=0.02, random_state=1234) + np.eye(npt)
    A0 = coo_matrix(fmat.transpose() * fmat)
    I0 = A0.row
    J0 = A0.col
    V0 = A0.data
    perm_rcm = reverse_cuthill_mckee(
        csr_matrix((V0, (I0, J0)), shape=(npt, npt)), symmetric_mode=True
    ).astype(np.int32)
    iperm_rcm = np.zeros(shape=npt, dtype=np.int32)
    for js, jt in enumerate(perm_rcm):
        iperm_rcm[jt] = js
    assert np.all(perm_rcm[iperm_rcm] == np.array(range(perm_rcm.size)))
    I_rcm = iperm_rcm[I0].astype(I0.dtype)
    J_rcm = iperm_rcm[J0].astype(J0.dtype)
    V_rcm = V0
    A_rcm = coo_matrix((V_rcm, (I_rcm, J_rcm)), shape=(npt, npt))
    A2_rcm = A0.copy()
    A2_rcm = A2_rcm.tocsr()[perm_rcm[:, None], perm_rcm]
    A2_rcm = coo_matrix(A2_rcm)

    mks = 2.0
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.spy(A0, markersize=mks)
    ax1.set_title("Before permutation", fontweight="bold")
    ax1.set_xticks([0, npt // 2, npt])
    ax1.set_yticks([0, npt // 2, npt])

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.spy(A_rcm, markersize=mks)
    ax2.set_xticks([0, npt // 2, npt])
    ax2.set_yticks([0, npt // 2, npt])
    ax2.set_title("After permutation (coo)", fontweight="bold")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.spy(A2_rcm, markersize=mks)
    ax3.set_xticks([0, npt // 2, npt])
    ax3.set_yticks([0, npt // 2, npt])
    ax3.set_title("After permutation (csr)", fontweight="bold")

    plt.show()


def mm_matrix(name):
    from scipy import sparse
    from scipy.io import mmread

    # Supposedly, it is better to use resource_stream and pass the resulting
    # open file object to mmread()... but for some reason this fails?
    from pkg_resources import resource_filename

    filename = resource_filename(__name__, "../data/sksparse/%s.mtx.gz" % name)
    matrix = mmread(filename)
    if sparse.issparse(matrix):
        matrix = matrix.tocsc()
    return matrix


def test_cholesky_matrix_market():
    # from functools import partial
    import numpy as np

    # from numpy.testing import assert_allclose
    from sksparse.cholmod import (
        # cholesky,
        cholesky_AAt,
        _modes,
    )
    import matplotlib.pyplot as plt

    modes = tuple(_modes.keys())
    # Match defaults of np.allclose, which were used before (and are needed).
    # assert_allclose = partial(assert_allclose, rtol=1e-5, atol=1e-8)
    # Adapted from https://github.com/scikit-sparse/scikit-sparse/blob/master/tests/test_cholmod.py
    for problem in ("well1033", "illc1033", "well1850", "illc1850"):
        X = mm_matrix(problem)
        y = mm_matrix(problem + "_rhs1")
        answer = np.linalg.lstsq(X.todense(), y)[0]
        XtX = (X.T * X).tocsc()
        Xty = X.T * y
        for mode in modes:
            # assert_allclose(cholesky(XtX, mode=mode)(Xty), answer)
            # assert_allclose(cholesky_AAt(X.T, mode=mode)(Xty), answer)
            # assert_allclose(cholesky(XtX, mode=mode).solve_A(Xty), answer)
            # assert_allclose(cholesky_AAt(X.T, mode=mode).solve_A(Xty), answer)
            factor = cholesky_AAt(X.T, mode=mode)
            permute = factor.P()
            X_p = X[:, permute]
            fig = plt.figure()
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.spy(X, markersize=1)
            ax1.set_xticks([0, X.shape[1] // 2, X.shape[1]])
            ax1.set_yticks([0, X.shape[0] // 2, X.shape[0]])
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.spy(X_p, markersize=1)
            ax2.set_xticks([0, X.shape[1] // 2, X.shape[1]])
            ax2.set_yticks([0, X.shape[0] // 2, X.shape[0]])
            plt.show()
