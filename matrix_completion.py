#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 03:47:45 2020
@author: Maryam
"""

import numpy as np
import argparse
import pickle
import time

from scipy.sparse.linalg import svds

from utils.read_preprocss_data import read_preprocss_data


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
parser.add_argument('--n_epochs', type=int, default=100,
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
    # # use a subset of data just for testing everything first
    # nu=100 # number of users
    # ni=200 # number of items
    # A_temp = A.copy()
    # data = A_temp[:nu,:ni] # small 10 X 20 submatrix
    # print(data.shape)
    #
    # A_temp = A_fill_zeros.copy()
    # data_fill_zeros = A_temp[:nu,:ni] # small 10 X 20 submatrix
    # print(data_fill_zeros.shape)
    data = A.copy()
    data_fill_zeros = A_fill_zeros.copy()
    print('data shape is:', data.shape)
    print('data fill zero shape is:', data_fill_zeros.shape)
    #===========================================================================
    zero_nums = (np.sum((data_fill_zeros==0).astype(int)))
    nonzero_nums = (np.sum((data_fill_zeros!=0).astype(int)))
    sparsity = zero_nums / (zero_nums+nonzero_nums)
    print('sparsity index of the data is', sparsity)
    #===========================================================================
    # STEP
    #===========================================================================
    n_k = [2, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # n_k = [2, 10, 15, 20, 30]
    MSEs_train = np.zeros((args.n_epochs, len(n_k)))
    MSEs_test = np.zeros((args.n_epochs, len(n_k)))
    RMSEs_test = np.zeros((args.n_epochs, len(n_k)))

    inds=np.nonzero(data_fill_zeros)
    nn=inds[0].shape[0]
    num_test = np.ceil(args.test_prc*nn).astype(int)

    for epch in range(args.n_epochs):

        print('-------------\nEpochs %s starts\n-------------' %epch)
        ir = np.random.permutation(nn)

        inds0 = inds[0].copy()
        inds1 = inds[1].copy()

        tst_ind0 = np.asarray([inds0[ir[i]] for i in range(num_test)])
        tst_ind1 = np.asarray([inds1[ir[i]] for i in range(num_test)])

        tst_trget = data[tst_ind0, tst_ind1].copy()
        train_data = data.copy()
        train_data[tst_ind0, tst_ind1] = 0

        for ikk, kk in enumerate(n_k):
                time_start=time.time()
                print('k: ', kk)
                print('ikk:', ikk)

                U, sigmaTmp, Vt = svds(data, k = kk)
                sigma = np.zeros([sigmaTmp.shape[0], sigmaTmp.shape[0]])
                np.fill_diagonal(sigma, sigmaTmp)
                pred_ratings = np.dot(np.dot(U, sigma), Vt)
                print('pred_ratings time elapsed: {} sec'.format(time.time()-time_start))

                err = (pred_ratings[tst_ind0, tst_ind1] - tst_trget)**2
                MSE = np.mean(err)
                RMSE = np.sqrt(MSE)
                MSEs_test[epch, ikk] = MSE
                RMSEs_test[epch, ikk] = RMSE
                print('MSE is:', MSE)
                print('RMSE is:', RMSE)
                if epch%5==0:
                    # Save errors
                    fn_str = args.RESULTPATH + 'mc_MSE_epch%s.npy' %(epch)
                    with open(fn_str, 'wb') as f:
                        pickle.dump(MSEs_test, f)
                    fn_str = args.RESULTPATH + 'mc_RMSE_epch%s.npy' %(epch)
                    with open(fn_str, 'wb') as f:
                        pickle.dump(RMSEs_test, f)
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
    print('DONE!!!')
