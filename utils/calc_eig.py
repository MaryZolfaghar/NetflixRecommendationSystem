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

    e, v = np.linalg.eig(L)
    v = v.real
    # eigenvalues
    print('eigenvalues:')
    print(e.shape)
    # eigenvectors
    print('eigenvectors:')
    print(v.shape)
    if args.normalize_laplacian:
        Y = np.sort(e)
        I = np.argsort(e)
        v_norm = v[:,I[:args.norm_laplacian_k]] \
            / LA.norm(v[:,I[:args.norm_laplacian_k]])*vol**(1/2)
    else:
        v_norm = []
    return e, v, v_norm
