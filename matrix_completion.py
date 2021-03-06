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
                    default='cosine_similarity',
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
parser.add_argument('--graph_nodes', choices=['M','U'],
                    default='M',
                    help='the nodes to create graph was either movies or users')

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
    n_k = [4, 5]

    MSEs_train = np.zeros((args.n_epochs, len(n_k)))
    RMSEs_train = np.zeros((args.n_epochs, len(n_k)))
    MSEs_test = np.zeros((args.n_epochs, len(n_k)))
    RMSEs_test = np.zeros((args.n_epochs, len(n_k)))

    counts_corr_train = np.zeros((args.n_epochs, len(n_k)))
    counts_corr_test = np.zeros((args.n_epochs, len(n_k)))
    prc_correct_train = np.zeros((args.n_epochs, len(n_k)))
    prc_correct_test = np.zeros((args.n_epochs, len(n_k)))

    inds=np.nonzero(data_fill_zeros)
    nn=inds[0].shape[0]
    num_test = np.ceil(args.test_prc*nn).astype(int)
    num_train = nn-num_test

    for epch in range(args.n_epochs):

        print('-------------\nEpochs %s starts\n-------------' %epch)
        ir = np.random.permutation(nn)

        inds0 = inds[0].copy()
        inds1 = inds[1].copy()

        tst_ind0 = np.asarray([inds0[ir[i]] for i in range(num_test)])
        tst_ind1 = np.asarray([inds1[ir[i]] for i in range(num_test)])

        tr_ind0 = np.asarray([inds0[ir[i+num_test]] for i in range(num_train)])
        tr_ind1 = np.asarray([inds1[ir[i+num_test]] for i in range(num_train)])

        tst_trget = data[tst_ind0, tst_ind1].copy()
        train_data = data.copy()
        print('train_data.shape', train_data.shape)
        train_data[tst_ind0, tst_ind1] = 0
        trn_trget = train_data[tr_ind0, tr_ind1].copy()

        for ikk, kk in enumerate(n_k):
                time_start=time.time()
                print('k: ', kk)
                print('ikk:', ikk)

                U, sigmaTmp, Vt = svds(train_data, k = kk)
                sigma = np.zeros([sigmaTmp.shape[0], sigmaTmp.shape[0]])
                np.fill_diagonal(sigma, sigmaTmp)
                pred_ratings = np.dot(np.dot(U, sigma), Vt)
                print('pred_ratings time elapsed: {} sec'.format(time.time()-time_start))

                err_tr = (pred_ratings[tr_ind0, tr_ind1] - trn_trget)**2
                err_ts = (pred_ratings[tst_ind0, tst_ind1] - tst_trget)**2

                diff_tr = (pred_ratings[tr_ind0, tr_ind1] - trn_trget)
                incorrect_tr = np.nonzero(diff_tr)[0]
                count_correct_tr = diff_tr.shape[0] - incorrect_tr.shape[0]
                prc_correct_tr = count_correct_tr/diff_tr.shape[0]
                counts_corr_train[epch, ikk] = count_correct_tr
                prc_correct_train[epch, ikk] = prc_correct_tr
                print('count correct train ', count_correct_tr)
                print('percentage correct train ', prc_correct_tr)


                diff_ts = (pred_ratings[tst_ind0, tst_ind1] - tst_trget)
                incorrect_ts = np.nonzero(diff_ts)[0]
                count_correct_ts = diff_ts.shape[0] - incorrect_ts.shape[0]
                prc_correct_ts = count_correct_ts/diff_ts.shape[0]
                counts_corr_test[epch, ikk] = count_correct_ts
                prc_correct_test[epch, ikk] = prc_correct_ts
                print('count correct test ', count_correct_tr)
                print('percentage correct test ', prc_correct_tr)



                MSE_tr = np.mean(err_tr)
                RMSE_tr = np.sqrt(MSE_tr)
                MSEs_train[epch, ikk] = MSE_tr
                RMSEs_train[epch, ikk] = RMSE_tr
                print('MSE train is:', MSE_tr)
                print('RMSE train is:', RMSE_tr)

                MSE_ts = np.mean(err_ts)
                RMSE_ts = np.sqrt(MSE_ts)
                MSEs_test[epch, ikk] = MSE_ts
                RMSEs_test[epch, ikk] = RMSE_ts
                print('MSE test is:', MSE_ts)
                print('RMSE test is:', RMSE_ts)
                if epch%50==0:
                    fn_str = args.RESULTPATH + 'mc_pred_rating_%s_%s_%s_epch%s.npy' \
                    %(args.fillnan, args.sim_method, args.test_prc, epch)
                    with open(fn_str, 'wb') as f:
                        pickle.dump(pred_ratings, f)

                    # Save errors
                    fn_str = args.RESULTPATH + 'mc_MSE_tr_%s_%s_%s_epch%s.npy' \
                    %(args.fillnan, args.sim_method, args.test_prc, epch)
                    with open(fn_str, 'wb') as f:
                        pickle.dump(MSEs_train, f)

                    fn_str = args.RESULTPATH + 'mc_RMSE_tr_%s_%s_%s_epch%s.npy' \
                    %(args.fillnan, args.sim_method, args.test_prc, epch)
                    with open(fn_str, 'wb') as f:
                        pickle.dump(RMSEs_train, f)

                    fn_str = args.RESULTPATH + 'mc_MSE_ts_%s_%s_%s_epch%s.npy' \
                    %(args.fillnan, args.sim_method, args.test_prc, epch)
                    with open(fn_str, 'wb') as f:
                        pickle.dump(MSEs_test, f)

                    fn_str = args.RESULTPATH + 'mc_RMSE_ts_%s_%s_%s_epch%s.npy' \
                    %(args.fillnan, args.sim_method, args.test_prc, epch)
                    with open(fn_str, 'wb') as f:
                        pickle.dump(RMSEs_test, f)

                    #
                    fn_str = args.RESULTPATH + 'mc_cnt_corr_tr_%s_%s_%s_epch%s.npy' \
                    %(args.fillnan, args.sim_method, args.test_prc, epch)
                    with open(fn_str, 'wb') as f:
                        pickle.dump(counts_corr_train, f)

                    fn_str = args.RESULTPATH + 'mc_cnt_corr_ts_%s_%s_%s_epch%s.npy' \
                    %(args.fillnan, args.sim_method, args.test_prc, epch)
                    with open(fn_str, 'wb') as f:
                        pickle.dump(counts_corr_test, f)

                    fn_str = args.RESULTPATH + 'mc_prc_corr_tr_%s_%s_%s_epch%s.npy' \
                    %(args.fillnan, args.sim_method, args.test_prc, epch)
                    with open(fn_str, 'wb') as f:
                        pickle.dump(prc_correct_train, f)

                    fn_str = args.RESULTPATH + 'mc_prc_corr_ts_%s_%s_%s_epch%s.npy' \
                    %(args.fillnan, args.sim_method, args.test_prc, epch)
                    with open(fn_str, 'wb') as f:
                        pickle.dump(prc_correct_test, f)






                    print('saving in matrix completion is done')
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
