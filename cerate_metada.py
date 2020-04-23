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
    print('saving in creating metadata is done')

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
