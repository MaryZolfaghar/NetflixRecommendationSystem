"""
Calculate eigen vectors and values of the input
"""

import numpy as np
import time

from numpy import linalg as LA
from scipy.sparse import linalg
# from scipy.linalg import eig as LAeig

def calc_eig(args, L, Ws):

    D=np.diag(np.sum(Ws, axis=0))
    vol=np.sum(np.diag(D))

    vals, vecs = np.linalg.eig(L)
    vecs = vecs.real
    # eigenvalues
    print('eigenvalues:')
    print(vals.shape)
    # eigenvectors
    print('eigenvectors:')
    print(vecs.shape)
    if args.normalize_laplacian:
        Y = np.sort(vals)
        I = np.argsort(vals)
        v_norm = vecs[:,I[:args.norm_laplacian_k]] \
            / LA.norm(vecs[:,I[:args.norm_laplacian_k]])*vol**(1/2)
    else:
        v_norm = []
    return vals, vecs, v_norm
