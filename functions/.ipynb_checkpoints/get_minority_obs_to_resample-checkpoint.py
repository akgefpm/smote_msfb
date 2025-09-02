from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_predict
import numpy as np

def get_minority_obs_to_resample(
    x_train, y_train,
    config):
    """    
    Identify minority samples to resample by:
      1) Creating a bagged classification model. Classifier could be supplied by the user. Default is Logistic Regression
      2) Cross-validated misclassification
      3) Filtering out 'noise' points with lower-than-threshold proba
    
    Returns:
        Tuple of:
        - List of minority-class indices to target for oversampling
        - Corresponding list of predicted probabilities for those indices
    """
        
    print(type(config))
    class_algo = eval(config["get_minority_obs_to_resample"]["classification_model"])
    
    print(class_algo)
    
    ## Create a bagged classifier based on the classification model supplied by user
    clf = BaggingClassifier(
        base_estimator=class_algo,
        n_estimators= config["get_minority_obs_to_resample"]["n_estimators"],
        max_samples=0.8,
        max_features=0.8,
        random_state= config["random_state"],
        n_jobs=-1
    )

    # Cross-validated probabilities    
    y_proba_cv = cross_val_predict(
        clf, x_train, y_train, 
        cv=config["get_minority_obs_to_resample"]["cv"],
        method='predict_proba', 
        n_jobs=-1
    )[:, 1]  # Probability of class 1

    # Find minority class samples
    minority_indices = np.where(y_train == 1)[0]

    # Identify misclassified minority samples
    miscls = [i for i in minority_indices if y_proba_cv[i] <= 0.5]

    # Filter by predicted probability
    filtered = [i for i in miscls if y_proba_cv[i] >= config["get_minority_obs_to_resample"]["prob_threshold"] ]

    # Fallback if too few filtered
    if len(filtered) / len(minority_indices) < config["get_minority_obs_to_resample"]["min_fraction"]:
        print(f"Fewer than {config['get_minority_obs_to_resample']['min_fraction']} samples after filtering with threshold {config['get_minority_obs_to_resample']['prob_threshold']}. Lowering to {config['get_minority_obs_to_resample']['fallback_threshold']}.")
        filtered = [i for i in miscls if y_proba_cv[i] >= config["get_minority_obs_to_resample"]["fallback_threshold"]]

    # Reporting
    print(f"Total minority samples: {len(minority_indices)}")
    print(f"Misclassified by bagged classification models: {len(miscls)}")
    print(f"After filtering proba < {config['get_minority_obs_to_resample']['prob_threshold']:.2f}: {len(filtered)} remain")

    # Also return probabilities for filtered samples
    filtered_probs = [y_proba_cv[i] for i in filtered]

    return filtered, filtered_probs
