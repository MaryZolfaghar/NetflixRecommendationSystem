"""
Generating similarity matrix
"""

import numpy as np
import pickle
from sklearn.metrics.pairwise import sigmoid_kernel, cosine_similarity


def gen_similarity(args, X):

    if args.sim_method=='sigmoid_kernel':
        sim_UXU=sigmoid_kernel(X=X, Y=None, gamma=None, coef0=1)
        sim_MXM=sigmoid_kernel(X=X.T, Y=None, gamma=None, coef0=1)
    elif args.sim_method=='cosine_similarity':
        sim_UXU=cosine_similarity(X=X, Y=None)
        sim_MXM=cosine_similarity(X=X.T, Y=None)
    ## =====================================================================
    # Save similarity matrix
    fn_str = args.RESULTPATH + 'sim_%s_UXU.npy' %(args.sim_method)
    with open(fn_str, 'wb') as f:
        pickle.dump(sim_UXU, f)

    fn_str = args.RESULTPATH + 'sim_%s_MXM.npy' %(args.sim_method)
    with open(fn_str, 'wb') as f:
        pickle.dump(sim_MXM, f)
    print('saving similarity matrix is done!')
    ## =====================================================================
    return sim_UXU, sim_MXM
