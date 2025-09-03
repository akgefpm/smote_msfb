
import numpy as np
from sklearn.neighbors import NearestNeighbors

def get_hubness_score_minority_class(x_train, y_train, weights,  k=5):
    """
    Compute normalized bad-hubness scores for minority class observations.
    Normalization is done so that the scores across all minority samples sum to 1.
    
    Parameters
    ----------
    x_train : array-like, shape (n_samples, n_features)
        Training data (binary feature space).
    y_train : array-like, shape (n_samples,)
        Class labels (0 = majority, 1 = minority).
    k : int
        Number of neighbors for k-NN (default=5).
        
    Returns
    -------
    bad_hub_scores : dict
        Mapping {minority_index: normalized_bad_hub_score}.
        Scores sum to 1 across all minority samples.
    """
    
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    # k-NN model with Jaccard distance
    nbrs = NearestNeighbors(n_neighbors=k+1, metric="jaccard")  
    nbrs.fit(x_train)
    neighbor_indices = nbrs.kneighbors(return_distance=False)

    n_samples = x_train.shape[0]
    
    # Counters
    bad_hub_count = np.zeros(n_samples, dtype=int)

    # Count hubness
    for i in range(n_samples):
        # Exclude self (first neighbor is always itself)
        for neigh in neighbor_indices[i][1:]:
            if y_train[i] != y_train[neigh]:
                # Neighbor is different class → bad hub for "neigh"
                bad_hub_count[neigh] += 1

    # Focus only on minority samples
    minority_indices = np.where(y_train == 1)[0]
    minority_bad_counts = bad_hub_count[minority_indices]

    # Normalize scores (sum to 1)
    total_bad = minority_bad_counts.sum()
    if total_bad > 0:
        normalized_scores = minority_bad_counts / total_bad
    else:
        # no bad hubs at all → assign uniform zero
        normalized_scores = np.zeros_like(minority_bad_counts, dtype=float)

    # Return as dictionary {index: score}
    bad_hub_scores = {
        idx: score for idx, score in zip(minority_indices, normalized_scores)
    }

    return bad_hub_scores
