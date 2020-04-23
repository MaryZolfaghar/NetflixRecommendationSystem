"""
Calculate eigen vectors and values of the input
"""

import numpy as np
import time

# from numpy import linalg as LA
# from scipy.sparse import linalg
# from scipy.linalg import eig as LAeig
# from scipy import linalg
from scipy.sparse.linalg import eigsh


def calc_eig(args, L, Ws, kk):
    t1 = time.time()
    D=np.diag(np.sum(Ws, axis=0))
    vol=np.sum(np.diag(D))

    # vals, vecs = np.linalg.eig(L)
    # vals, vecs = linalg.eig(L)
    vals, vecs = eigsh(L, k=kk, which="SM")  # Largest 5 eigenvalues/vectors
    vecs = vecs.real
    # eigenvalues
    print('eigenvalues:')
    print(vals.shape)
    # eigenvectors
    print('eigenvectors:')
    print(vecs.shape)
    if args.normalize_laplacian:
        print('do the normalization')
        Y = np.sort(vals)
        I = np.argsort(vals)
        v_norm = vecs[:,I[:args.norm_laplacian_k]] \
            / LA.norm(vecs[:,I[:args.norm_laplacian_k]])*vol**(1/2)
    else:
        v_norm = []
    elapsed_time = time.time() - t1
    print('Elapsed time is {} seconds: '.format(elapsed_time))
    print('calc eigen vectors and values done!')
    return vals, vecs, v_norm
