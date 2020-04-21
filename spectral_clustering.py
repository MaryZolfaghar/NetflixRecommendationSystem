#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 03:47:45 2020
@author: Maryam
"""

import numpy as np
import argparse
import pickle

from sklearn.cluster import KMeans

from utils.read_preprocss_data import read_preprocss_data
from utils.calc_eig import calc_eig
from utils.calc_laplacian import calc_laplacian
from utils.gen_similarity import gen_similarity

parser = argparse.ArgumentParser()

# Set Path
parser.add_argument("--DATAPATH",
                    default='../datasets/',
                    help='Filename for datasets')
parser.add_argument("--RESULTPATH",
                    default='../results/',
                    help='Filename for saving the results')
# Preprocessing
parser.add_argument('--metadata', action='store_true',
                    help='whether use metadata or not')
parser.add_argument('--fillnan', choices=['mean_row','mean_col'],
                    default='mean_col',
                    help='Whether fill NaN with the mean of rows or columns')

# Similarity
parser.add_argument('--sim_method', choices=['sigmoid_kernel','cosine_similarity'],
                    default='sigmoid_kernel',
                    help='What type of similarity method should use')
# Spectral clustering
parser.add_argument('--norm_laplacian_k', type=int, default=5,
                    help='k in laplacian normalization and its eigen vector clustering')
parser.add_argument('--normalize_laplacian', action='store_true',
                    help='whether normalize laplacian or not')
# Kmeans
parser.add_argument('--kmeans_k', type=int, default=5,
                    help='number of clusters in kmeans')

# train
parser.add_argument('--n_epochs', type=int, default=10,
                    help='number of epochs')
parser.add_argument('--test_prc', type=float, default=0.1,
                    help='percentage for test dataset')

"""
main function
"""
def main(args):
    df, A, A_fill_zeros = read_preprocss_data(args)
    print('done reading the data')

    #===========================================================================
    # use a subset of data just for testing everything first
    nu=100 # number of users
    ni=200 # number of items
    A_temp = A.copy()
    data = A_temp[:nu,:ni] # small 10 X 20 submatrix
    print(data.shape)
    #===========================================================================
    zero_nums = (np.sum((data==0).astype(int)))
    nonzero_nums = (np.sum((data!=0).astype(int)))
    sparsity = zero_nums / (zero_nums+nonzero_nums)
    print('sparsity index of the data is', sparsity)
    #===========================================================================
    # STEP 1 - Calculate similarity
    #===========================================================================
    sim_UXU, sim_MXM = gen_similarity(args, data)
    print('gen similarity is done')
    #===========================================================================
    # STEP 2 - computing the laplacian
    # If the graph (W) has K connected components, then L has K eigenvectors
    #with an eigenvalue of 0.
    #===========================================================================
    Ws = sim_MXM.copy()
    L, D = calc_laplacian(args, Ws)
    print('calc laplacian is done')
    #===========================================================================
    # STEP 3 - Compute the eigenvectors of the matrix L
    #===========================================================================
    e, v, v_norm = calc_eig(args, L, Ws)
    print('calc eigens is done')
    #===========================================================================
    # STEP 4 - Using the k smallest eigenvector as input,
    # train a k-means model and use it to classify the data
    #===========================================================================
    U = np.array(v)
    km = KMeans(init='k-means++', n_clusters=args.kmeans_k)
    km.fit(U)
    print(km.labels_.shape)
    print('calc kmeans is done')
    # Save labels
    fn_str = args.RESULTPATH + 'kmeans_obj_MXM_k%s_%s' %(args.kmeans_k, args.sim_method)
    with open(fn_str, 'wb') as f:
        pickle.dump(km, f)
    print('saving kmenas is done')
    #===========================================================================
    # STEP 5 - using k centers to predict data
    #===========================================================================

"""
==============================================================================
Main
==============================================================================
"""
if __name__ == '__main__':
    args=parser.parse_args()
    print('-------Arguments:---------')
    print(args)
    print('--------------------------')
    main(args)
