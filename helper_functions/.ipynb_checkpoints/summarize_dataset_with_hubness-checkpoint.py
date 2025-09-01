import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.stats import skew

from compute_hubness_metrics import compute_hubness_metrics

def summarize_dataset_with_hubness(data, feature_prefix="f_", target_prefix="target_", k=10):
    """
    Summarize dataset with hubness metrics for each target_* variable.
    """
    features = [c for c in data.columns if c.startswith(feature_prefix)]
    targets = [c for c in data.columns if c.startswith(target_prefix)]

    results = []

    for target in targets:
        y = data[target].values
        X = data[features].values

        # hubness metrics
        hubness = compute_hubness_metrics(X, y, k=k)

        # imbalance stats
        counts = pd.Series(y).value_counts()
        maj, mino = counts.max(), counts.min()
        imbalance_ratio = maj / mino if mino > 0 else np.inf

        result = {
            "target": target,
            "n_samples": len(y),
            "n_features": len(features),
            "imbalance_ratio": imbalance_ratio,
            "n_classes": counts.shape[0],
            "n_majority_samples": maj,
            "n_minority_samples": mino            
        }
        result.update(hubness)
        results.append(result)

    return pd.DataFrame(results)