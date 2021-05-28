import numpy as np

def solve(h,V):
    """
    :param h: the kinetic + external energy matrices
    :param V: the interaction energy tensor
    :return vals: the eigenvalues of the full Hamiltonian of the system
    :return vecs: the eigenvectors of the full Hamiltonian of the system
    """

    # save the leading dimension of the matrix h as it is the number of sites
    N = h.shape[0]

    # Transform the 4 dimensional tensor V into and (N*N,N*N) matrix
    oneDlist = []
    for ii in range(N):
        for kk in range(N):
            for jj in range(N):
                for ll in range(N):
                    oneDlist.append(V[ii, jj, kk, ll])
    Vmatrix = np.array(oneDlist).reshape((N * N, N * N))

    # the matrix h needs to be transformed so that it is in the same basis as the matrix Vmatrix
    # then the eigenvalues and vectors of the system can be solved.
    ham = np.kron(h, np.identity(N)) + np.kron(np.identity(N), h) + 2 * Vmatrix
    vals, vecs = np.linalg.eigh(ham)

    return vals, vecs