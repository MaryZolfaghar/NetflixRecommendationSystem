"""
Plotting high-dimension data using tSNE
"""
import numpy as np
import time
# from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

def tSNE_plot(args, data, labels):
    print('data shape:', data.shape)

    time_start = time.time()

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data)
    xx = tsne_results[:,0]
    yy = tsne_results[:,1]

    plt.figure(figsize=(14,10))
    sns.scatterplot(
        x=xx, y=yy,
        hue=labels,
        palette=sns.color_palette("hls", args.kmeans_k),
        legend="full",
        alpha=0.3
    )
