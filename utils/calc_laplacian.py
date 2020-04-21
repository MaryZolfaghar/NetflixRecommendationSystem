"""
Calculating the Laplacian matrix
"""

import numpy as np
import time

def calc_laplacian(args, Ws):
    t1 = time.time()
    # degree matrix
    D = np.diag(np.sum(np.array(Ws), axis=1))
    print('degree matrix:')
    print(D.shape)
    # laplacian matrix
    L = D - Ws
    print('laplacian matrix:')
    print(L.shape)
    elapsed_time = time.time() - t1
    print('Elapsed time is {} seconds: '.format(elapsed_time))

    return L, D
