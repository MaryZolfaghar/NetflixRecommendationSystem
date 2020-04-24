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

from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering

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
parser.add_argument('--n_epochs', type=int, default=10,
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
    # use a subset of data just for testing everything first
    # nu=100 # number of users
    # ni=200 # number of items
    # A_temp = A.copy()
    # data = A_temp[:nu,:ni] # small 10 X 20 submatrix
    # print(data.shape)
    #
    # A_temp = A_fill_zeros.copy()
    # data_fill_zeros = A_temp[:nu,:ni] # small 10 X 20 submatrix

    data = A.copy()
    train_data = data.copy()
    data_fill_zeros = A_fill_zeros.copy()
    print('data shape is:', data.shape)
    print('data fill zero shape is:', data_fill_zeros.shape)
    #===========================================================================
    #=======================================================================
    test = pd.read_csv(args.DATAPATH + 'test.csv')
    test.columns  = ['movie_id', 'customer_id', 'rating', 'date']
    test_np = test.to_numpy().copy()

    train = pd.read_csv(args.DATAPATH + 'train.csv')
    train.columns  = ['movie_id', 'customer_id', 'rating', 'date']
    train_np = train.to_numpy().copy()

    #===========================================================================
    # STEP 4 - Using the k smallest eigenvector as input,
    # train a k-means model and use it to classify the data
    #===========================================================================
    if args.graph_nodes=='M':
        n_k = [10, 50, 100]
    elif args.graph_nodes=='U':
        n_k = [10, 50, 100]
    #=======================================================================
    final_k = 4
    #=======================================================================
    # STEP 1 - Calculate similarity
    sim_UXU, sim_MXM = gen_similarity(args, train_data)
    print('gen similarity is done')

    # STEP 2 - computing the laplacian
    if args.graph_nodes=='M':
        Ws = sim_MXM.copy()
    elif args.graph_nodes=='U':
        Ws = sim_UXU.copy()
    L, D = calc_laplacian(args, Ws)
    print('calc laplacian is done')

    # STEP 3 - Compute the eigenvectors of the matrix L
    vals, vecs, v_norm, eigengap = calc_eig(args, L, Ws, final_k)

    # STEP 5 - using k centers to predict data
    U = np.array(vecs)
    print('U array eigenvectors shape:', U.shape)

    t1=time.time()
    km = MiniBatchKMeans(n_clusters=final_k,
                         random_state=0,
                         batch_size=100,
                         max_iter=100)
    print('MiniBatchKMeans time elapsed: {} sec'.format(time.time()-t1))
    km.fit(U)
    print('MiniBatchKMeans Fit time elapsed: {} sec'.format(time.time()-t1))

    if args.graph_nodes=='M': # menas the sim is MXM
        labels = np.zeros([final_k])
        pred_ratings = np.zeros(train_data.shape[1])
        t0=time.time()
        for il, lbl in enumerate(range(kk)):
            print('label is:', lbl)
            print('km.labels_', np.unique(km.labels_))
            dfz=data_fill_zeros[:,km.labels_==lbl].copy()

            # find user that rated at least one of the movies
            goodU= np.mean(dfz, axis=1)
            if goodU.shape[0] > 0:
                # index for users that rate at least one of
                # the movies in that clustr
                indxgu=np.where(goodU > 0)
                trdata = train_data[:, km.labels_==lbl]
                trdata = trdata[indxgu[0], :]
            else:
                trdata = train_data[:, km.labels_==lbl]

            trdata = np.mean(trdata,axis=0)
            labels[il] = np.ceil(np.mean(trdata,axis=0))
#test -> DataFrame
#test_np -> numpy
#train -> DataFrame
#train_np -> numpy
#train_data -> col mean
#data_fill_zeros -> train with zero fill

        for ic in range(test_np.shape[0]):
            # ctst = km.labels_[ic]
            # pred_ratings[ic] = labels[ctst]
            mvid_ts = test_np[ic,0]
            indx = np.where(train_np[:,0]==test_np[ic,0])
            if indx.shape[0] > 1:
                print('we have more movies with this id%s:', mvid_ts)

            # serach if this existed in train before
            dd = data_fill_zeros[:,train_np[:,0]==test_np[ic,0]]
            dd2 = data_fill_zeros[:,indx]
            print('two way shoud give zero:', dd2-dd)
            existed_r = d[train_np[:,1]==test_np[ic,1],:]
            if existed_r > 0:
                test_np[ic,2]=existed_r
            ctst = km.labels_[indx]
            test_np[ic,2] = labels[ctst]




            if ic%100==0:
                print('interation for finding clusters (ic)', ic)
                print('Epalsed time: {} sec'.format(time.time()-t0))
                print('\n')

    elif args.graph_nodes=='U': # menas the sim is UXU
        labels = np.zeros([kk])
        pred_ratings = np.zeros(train_data.shape[0])
        t0=time.time()
        for il, lbl in enumerate(range(kk)):
            print('label is:', lbl)
            print('km.labels_', np.unique(km.labels_))
            trdata = train_data[km.labels_==lbl,:]
            trdata = np.mean(trdata,axis=1)
            labels[il] = np.ceil(np.mean(trdata, axis=0))
        for ic in range(train_data.shape[0]):
            ctst = km.labels_[ic]
            pred_ratings[ic] = labels[ctst]

        if epch%5==0:
            # Save errors
            fn_str = args.RESULTPATH + 'sc_MSE_tr_%s_%s_%s_%s_epch%s.npy' \
            %(args.graph_nodes, args.fillnan, args.sim_method, args.test_prc, epch)
            with open(fn_str, 'wb') as f:
                pickle.dump(MSEs_train, f)

        for tst in range(test_df.shape[0])

        qry=test[(test['movie_id']==11279) & (test['customer_id']==5858)]
        qry['rating']=1

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
