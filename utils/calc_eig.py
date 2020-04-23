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

    vals, vecs = eigsh(L, k=kk, which="SM")  # Largest 5 eigenvalues/vectors
    vecs = vecs.real

    print('the first 10 eigen values are:')
    print(vals[:10])
    print('\n')

    if (vals[0]==0):
        if vals[1] > 0:
            print('OOOPS the first eigen value was zero')
            vals = vals[1:]
            vecs = vecs[:,1:]
    if (vals[0]<1e-10):
        print('OOOPS the first eigen value was so small')
        vals = vals[1:]
        vecs = vecs[:,1:]

    #caluclate eigen gap
    e1 = np.zeros([vals.shape[0]+1])
    e2 = np.zeros([vals.shape[0]+1])
    print(e1.shape)
    e1[1:] = vals.copy()
    e2[:-1] = vals.copy()
    print('eigen gap is:')
    eigengap=(e2-e1)
    print(eigengap)
    print('the first 10 eigen values are:')
    print(vals[:10])
    print('\n')
    #


    # eigenvalues
    print('eigenvalues shape is:')
    print(vals.shape)
    # eigenvectors
    print('eigenvectors shape is :')
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
    return vals, vecs, v_norm, eigengap
