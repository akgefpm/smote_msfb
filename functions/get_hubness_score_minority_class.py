
import numpy as np
from sklearn.neighbors import NearestNeighbors
import sys

sys.path.append('/repos/smote_msfb/functions')
from weighted_jaccard_distance import weighted_jaccard_distance


def get_hubness_score_minority_class(x_train, y_train, weights, k=5):
    """
    Compute normalized bad hubness score for minority samples using
    weighted Jaccard distance with MI-based weights.

    Args:
        x_train: (n_samples, n_features) training data (binary features)
        y_train: (n_samples,) labels (0=majority, 1=minority)
        k: number of neighbors
        weights: (n_features,) normalized MI weights for features

    Returns:
        bad_hub_scores: dict {sample_index: normalized bad hubness score}
    """
    # Identify minority indices
    minority_indices = np.where(y_train == 1)[0]

    #print("Type of weights :", type(weights),"   values:", weights)
    
    # Compute weighted Jaccard distance matrix
    jaccard_dist = weighted_jaccard_distance(x_train, weights)

    # KNN with precomputed distance matrix
    knn = NearestNeighbors(n_neighbors=k+1, metric="precomputed")
    knn.fit(jaccard_dist)
    distances, indices = knn.kneighbors(jaccard_dist)

    # Initialize bad hub scores for minority samples only
    bad_hub_scores = {idx: 0 for idx in minority_indices}

    # Count bad hubness
    for i, neigh_indices in enumerate(indices):
        for n_idx in neigh_indices[1:]:  # exclude self
            # Bad hubness = different labels
            if y_train[i] != y_train[n_idx]:
                if n_idx in minority_indices:  # only track minority samples
                    bad_hub_scores[n_idx] += 1

    # Normalize scores so they sum to 1 across minority samples
    total_bad = sum(bad_hub_scores.values())
    if total_bad > 0:
        for key in bad_hub_scores:
            bad_hub_scores[key] /= total_bad

    return bad_hub_scores
