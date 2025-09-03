
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.feature_selection import mutual_info_classif    
import math

def weighted_jaccard_distance(X, weights):
    """
    Computes weighted Jaccard distance matrix for binary data.
    X: binary input (n_samples x n_features)
    weights: array of feature weights (length = n_features)
    """
    
    #print("Type of weights :", type(weights),"   values:", weights)
    
    n_samples = X.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))

    # Ensure X is binary boolean for logical operations
    X_bool = X.astype(bool)

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            a = X_bool[i]
            b = X_bool[j]
            
            #print("Type of a :", type(a),"   values:", a)
            #print("Type of b :", type(b),"   values:", b)
            #print("Type of weights :", type(weights),"   values:", weights)

            intersection = np.sum(weights * (a & b))
            union = np.sum(weights * (a | b))

            distance = 1 - (intersection / union) if union != 0 else 1.0
            dist_matrix[i, j] = distance
            dist_matrix[j, i] = distance

    return dist_matrix