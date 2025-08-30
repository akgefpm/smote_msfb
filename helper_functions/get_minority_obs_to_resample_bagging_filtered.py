from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_predict
import numpy as np

def get_minority_obs_to_resample_bagging_filtered(
    x_train, y_train,
    n_estimators=10,
    max_depth=5,
    cv=5,
    prob_threshold=0.1,
    fallback_threshold=0.05,
    random_state=42
):
    """
    Identify minority samples to resample by:
      1) Bagged decision trees (captures non-linear structure)
      2) Cross-validated misclassification
      3) Filtering out 'noise' points with lower-than-threshold proba
    
    Returns:
        Tuple of:
        - List of minority-class indices to target for oversampling
        - Corresponding list of predicted probabilities for those indices
    """

    base_lr = LogisticRegression(max_iter=1000, random_state=random_state)

    clf = BaggingClassifier(
        base_estimator=base_lr,
        n_estimators=n_estimators,
        max_samples=0.8,
        max_features=0.8,
        random_state=random_state,
        n_jobs=-1
    )

    # Cross-validated predictions and probabilities
    y_pred_cv = cross_val_predict(clf, x_train, y_train, cv=cv, n_jobs=-1)
    y_proba_cv = cross_val_predict(
        clf, x_train, y_train, cv=cv,
        method='predict_proba', n_jobs=-1
    )[:, 1]  # Probability of class 1

    # Find minority class samples
    minority_indices = np.where(y_train == 1)[0]

    # Identify misclassified minority samples
    miscls = [i for i in minority_indices if y_pred_cv[i] != 1]

    # Filter by predicted probability
    filtered = [i for i in miscls if y_proba_cv[i] >= prob_threshold]

    # Fallback if too few filtered
    if len(filtered) / len(minority_indices) < 0.05:
        print(f"Fewer than 5% samples after filtering with threshold {prob_threshold}. Lowering to {fallback_threshold}.")
        filtered = [i for i in miscls if y_proba_cv[i] >= fallback_threshold]

    # Reporting
    print(f"Total minority samples: {len(minority_indices)}")
    print(f"Misclassified by bagged logistic models: {len(miscls)}")
    print(f"After filtering proba < {prob_threshold:.2f}: {len(filtered)} remain")

    # Also return probabilities for filtered samples
    filtered_probs = [y_proba_cv[i] for i in filtered]

    return filtered, filtered_probs
