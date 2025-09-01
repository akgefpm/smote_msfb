import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.stats import skew

def compute_hubness_metrics(X, y, k=10):
    """
    Compute hubness metrics for a dataset (X, y).
    
    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    y : pd.Series or np.ndarray
        Target vector.
    k : int
        Number of neighbors for hubness calculation.
    
    Returns:
    --------
    dict of hubness metrics
    """
    
    #X = (X > 0).astype(bool)
    
    # kNN search
    nn = NearestNeighbors(n_neighbors=k+1, metric="jaccard")
    nn.fit(X)
    neigh_indices = nn.kneighbors(X, return_distance=False)[:, 1:]  # exclude self

    n = X.shape[0]
    Nk = np.zeros(n)   # total hubness
    GNk = np.zeros(n)  # good hubness
    BNk = np.zeros(n)  # bad hubness

    # Count occurrences
    for i in range(n):
        for j in neigh_indices[i]:
            Nk[j] += 1
            if y[i] == y[j]:
                GNk[j] += 1
            else:
                BNk[j] += 1

    # Metrics
    mean_hubness = Nk.mean()
    mean_GNk = GNk.mean()
    mean_BNk = BNk.mean()
    prop_good_gt_bad = np.mean(GNk > BNk)
    skewness = skew(Nk)
    max_GNk = GNk.max()
    max_BNk = BNk.max()
    hubness_ratio = np.sum(BNk > GNk) / np.sum(Nk > 0)

    return {
        "mean_hubness": mean_hubness,
        "mean_GNk": mean_GNk,
        "mean_BNk": mean_BNk,
        "prop_good_gt_bad": prop_good_gt_bad,
        "skewness": skewness,
        "max_GNk": max_GNk,
        "max_BNk": max_BNk,
        "hubness_ratio": hubness_ratio
    }