

import numpy as np
from sklearn.ensemble import IsolationForest

def filter_anomalous_minority_samples(x_train, y_train, contamination=0.03, random_state=42, features_subset=None):
    """
    Identify and remove anomalous minority class samples.

    Args:
        x_train: numpy array (n_samples, n_features), training features (binary)
        y_train: numpy array (n_samples,), labels (0=majority, 1=minority)
        contamination: float, fraction of minority samples to treat as anomalies
        random_state: int, for reproducibility
        features_subset: list or None, indices of features to use for anomaly detection
                         (if None, use all features)

    Returns:
        valid_minority_indices: np.array, indices of minority samples NOT flagged as anomalies
    """

    # Select minority samples
    minority_indices = np.where(y_train == 1)[0]
    x_minority = x_train[minority_indices]

    # Use only selected features if provided
    if features_subset is not None:
        x_minority = x_minority[:, features_subset]

    # Fit Isolation Forest on minority samples
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    iso.fit(x_minority)

    # Predict anomalies: -1 -> anomaly, 1 -> normal
    preds = iso.predict(x_minority)

    # Keep only normal points
    valid_minority_indices = minority_indices[preds == 1]
    
    print("Minority samples removed from anamoly detection :", minority_indices[preds != 1])

    return valid_minority_indices
