
"""
Reading and preprocessing data
"""
import numpy as np
import pandas as pd
import pickle

def read_preprocss_data(args):

    # args.DATAPATH = '../datasets/'
    train = pd.read_csv(args.DATAPATH + 'train.csv')
    test = pd.read_csv(args.DATAPATH + 'test.csv')

    train.columns = ['movie_id', 'customer_id', 'rating', 'date']
    test.columns  = ['movie_id', 'customer_id', 'rating', 'date']

    df = train.pivot_table(index='customer-id', \
                               columns='movie-id', values='rating', aggfunc=np.mean).fillna(0)
    A_fill_zeros = df.to_numpy().copy()

    if args.fillnan=='mean_col':
        df = train.pivot_table(index='customer-id', \
                               columns='movie-id', values='rating', aggfunc=np.mean)
        A = df.to_numpy().copy()
        # column mean
        col_mean = np.nanmean(A, axis = 0)
        col_mean = np.ceil(col_mean)
        # find indices where nan value is present
        inds = np.where(np.isnan(A))
        # replace inds with avg of column
        A[inds] = np.take(col_mean, inds[1])
    elif args.fillnan=='mean_row':
        df = train.pivot_table(index='customer-id', \
                               columns='movie-id', values='rating', aggfunc=np.mean)
        A = df.to_numpy().copy()
        # row mean
        row_mean = np.nanmean(A, axis = 1)
        row_mean = np.ceil(row_mean)
        # find indices where nan value is present
        inds = np.where(np.isnan(A))
        # replace inds with avg of column
        A[inds] = np.take(row_mean, inds[1])


    print('Reading is done, the shape of the data is:', A.shape)
    ## Reading metadata ========================================================
    if args.metadata:
        t1=[]; t2=[]; t3=[]
        with open(args.DATAPATH + 'movie_titles.txt', 'r',encoding="latin-1") as reading:
            for line in reading.readlines():
                tokens = line.split(",")
                t1.append(tokens[0])
                t2.append(tokens[1])
                t33 = tokens[2].split('\n')
                t3.append(t33[0])
        t1=np.asarray(t1)
        t1=t1[1:]
        t2=np.asarray(t2)
        t2=t2[1:]
        t3=np.asarray(t3)
        t3=t3[1:]
        titles = pd.read_fwf(args.DATAPATH + 'movie_titles.txt', delimiter= ',', \
                                   names = ["movie_id", "year_produced", "title"], encoding="ISO-8859-1")
        movie_titles = pd.DataFrame(titles[1:], columns=["movie_id", "year_produced", "title"])
        movie_titles['movie_id'] = t1
        movie_titles['year_produced'] = t2
        movie_titles['title'] = t3
        ## =====================================================================
        # Save movie titles
        fn_str = args.DATAPATH + 'movie_titles_df.csv'
        with open(fn_str, 'wb') as f:
            pickle.dump(movie_titles, f)
        ## =====================================================================
        # try fiding correspoding data in train to add metadata features to it
        trnp = train.to_numpy()
        mtnp = movie_titles.to_numpy()
        train_metadata = train.copy()
        train_metadata['title'] = np.zeros([train_metadata.shape[0]])
        train_metadata['title'] = 'NaN'
        train_metadata['year_produced'] = np.zeros([train_metadata.shape[0]])
        train_metadata['year_produced']  = 'NaN'

        # try to add movie titles to the train data, seems this takes long so skip for now
        titles=[]
        years=[]
        print(trnp.shape)
        for itr in range(trnp.shape[0]):
            if (itr%1000==0):
                print(itr)
            aa=trnp[itr,0]
            mm = movie_titles[movie_titles.movie_id == str(aa)]
            titles.append(mm.title.to_string())
            years.append(mm.year_produced.to_string())

        titles = np.asarray(titles)
        print(titles.shape)
        train_metadata['title'] = titles
        train_metadata['year_produced'] = years
        ## =====================================================================
        # Save movie titles
        fn_str = args.DATAPATH + 'train_metadata.csv'
        with open(fn_str, 'wb') as f:
            pickle.dump(train_metadata, f)

        return df, A, A_fill_zeros
