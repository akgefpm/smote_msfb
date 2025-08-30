import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import check_random_state
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import mutual_info_classif
from imblearn.metrics.pairwise import ValueDifferenceMetric
from scipy.stats import mode
from get_minority_obs_to_resample_bagging_filtered import get_minority_obs_to_resample_bagging_filtered

def smoten_select_feature(X, y, sampling_strategy, k_neighbors=6, top_feature_percent=20, random_state=None):
    random_state = check_random_state(random_state)

    # Step 1: Encode categorical features as integers
    encoder = OrdinalEncoder(dtype=np.int32)
    X_encoded = encoder.fit_transform(X)

    # Step 2: Compute feature importance using mutual information
    mi_scores = mutual_info_classif(X_encoded, y, discrete_features=True, random_state=random_state)

    # Step 3: Select top features
    num_features_to_keep = max(1, int((top_feature_percent / 100) * X_encoded.shape[1]))
    top_features_indices = np.argsort(mi_scores)[-num_features_to_keep:]

    # Step 4: Reduce X to selected features
    X_selected = X_encoded[:, top_features_indices]

    # Step 5: Compute Value Difference Metric (VDM)
    vdm = ValueDifferenceMetric(
        n_categories=[len(cat) for i, cat in enumerate(encoder.categories_) if i in top_features_indices]
    ).fit(X_selected, y)

    X_resampled = [X_encoded.copy()]
    y_resampled = [y.copy()]

    # Step 6: Fit Nearest Neighbors
    nn_model = NearestNeighbors(n_neighbors=k_neighbors, metric="precomputed")

    for class_sample, n_samples in sampling_strategy.items():
        print("Class sample:", class_sample)
        if n_samples == 0:
            continue

        # Get indices of current class
        target_class_indices = np.flatnonzero(y == class_sample)
        X_class = X_selected[target_class_indices]

        # Compute VDM distances
        X_class_dist = vdm.pairwise(X_class)
        nn_model.fit(X_class_dist)
        nn_indices = nn_model.kneighbors(X_class_dist, return_distance=False)[:, 1:]

        #print("neighbor_indices:", nn_indices)

        # Get filtered sample indices and probabilities
        sample_indices_global, sample_probs_global = get_minority_obs_to_resample_bagging_filtered(X_encoded, y)

        # Filter for current class
        filtered = [(i, p) for i, p in zip(sample_indices_global, sample_probs_global) if y[i] == class_sample]

        if not filtered:
            print("No valid sample indices found for class", class_sample)
            continue

        global_indices, global_probs = zip(*filtered)

        # Normalize probabilities to sum to 1
        prob_dist = np.array(global_probs) / np.sum(global_probs)

        # Remap to local index space
        index_map = {global_idx: local_idx for local_idx, global_idx in enumerate(target_class_indices)}
        local_indices_with_probs = [
            (index_map[i], p) for i, p in zip(global_indices, prob_dist) if i in index_map
        ]

        if not local_indices_with_probs:
            print("No mapped indices found after filtering for class", class_sample)
            continue

        local_indices, local_probs = zip(*local_indices_with_probs)
        local_probs = np.array(local_probs)
        local_probs /= local_probs.sum()  # normalize again just in case

        print(f"Sampling {n_samples} from {len(local_indices)} misclassified samples (weighted).")

        # Weighted random sampling with replacement
        sample_indices = random_state.choice(local_indices, size=n_samples, replace=True, p=local_probs)

        # Generate synthetic samples using neighbor majority vote
        X_new = np.squeeze(
            mode(X_encoded[target_class_indices][nn_indices[sample_indices]], axis=1)[0],
            axis=1
        )
        y_new = np.full(n_samples, fill_value=class_sample, dtype=y.dtype)

        X_new = encoder.inverse_transform(X_new)
        X_resampled.append(X_new)
        y_resampled.append(y_new)

    # Combine original and synthetic data
    X_resampled = np.vstack(X_resampled)
    y_resampled = np.hstack(y_resampled)

    return X_resampled, y_resampled
