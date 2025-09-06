import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import copy
import sys

sys.path.append('/repos/smote_msfb/functions')
from smote_msfb import smote_msfb 

def smote_msfb_cross_validation(X_train, y_train, config, classification_model, n_splits=4):
    """
    Run stratified cross-validation with smote_msfb and test different resampling ratios.

    Parameters
    ----------
    X_train : np.ndarray or pd.DataFrame
        Feature matrix (binary high-dimensional covariates).
    y_train : np.ndarray or pd.Series
        Binary response variable (imbalanced).
    config : dict
        Configuration dictionary for smote_msfb.
    classification_model : sklearn-like estimator
        Model with .fit() and .predict_proba() methods.
    n_splits : int
        Number of stratified folds (default=4).

    Returns
    -------
    best_strategy : float
        The sampling_strategy value that gave the best mean ROC-AUC.
    results : dict
        Keys are sampling_strategy values, values are mean ROC-AUC across folds.
    """

    # Ensure numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Compute current minority/majority ratio
    minority_count = np.sum(y_train == 1)
    majority_count = np.sum(y_train == 0)
    current_ratio = minority_count / majority_count

    # Round up to nearest 0.1 and start from there
    start_ratio = np.ceil(current_ratio * 10) / 10
    if start_ratio < 0.1:
        start_ratio = 0.1  # safeguard for extreme imbalance

    sampling_strategies = np.round(np.arange(start_ratio, 1.1, 0.1), 2)

    print(f"\nCurrent minority/majority ratio ≈ {current_ratio:.2f}")
    print(f"Testing sampling_strategy values: {sampling_strategies}")

    # Storage for results
    results = {s: [] for s in sampling_strategies}

    # Stratified CV
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_train, y_train), 1):
        print(f"\n▶ Fold {fold_idx}/{n_splits}")

        X_tr, X_te = X_train[train_idx], X_train[test_idx]
        y_tr, y_te = y_train[train_idx], y_train[test_idx]

        for s in sampling_strategies:
            # Make a deep copy of config (to not overwrite across runs)
            cfg = copy.deepcopy(config)
            cfg['main_section']['sampling_strategy'] = s

            # Resample training data
            X_res, y_res = smote_msfb(X_tr, y_tr, cfg)

            # Train model
            model = copy.deepcopy(classification_model)
            model.fit(X_res, y_res)

            # Predict probabilities on test fold
            y_pred_prob = model.predict_proba(X_te)[:, 1]

            # Compute ROC-AUC
            auc = roc_auc_score(y_te, y_pred_prob)
            results[s].append(auc)
            print(f"  - sampling_strategy={s:.1f}, ROC-AUC={auc:.4f}")

    # Average across folds
    results = {s: np.mean(aucs) for s, aucs in results.items()}

    # Select best sampling strategy
    best_strategy = max(results, key=results.get)

    print("\n✅ Final mean ROC-AUC per sampling_strategy:")
    for s, auc in results.items():
        print(f"  {s:.1f}: {auc:.4f}")
    print(f"\n🏆 Best sampling_strategy: {best_strategy:.1f} with ROC-AUC={results[best_strategy]:.4f}")

    return best_strategy
