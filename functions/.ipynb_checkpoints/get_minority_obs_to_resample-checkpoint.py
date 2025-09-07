from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_predict
import numpy as np
import sys

sys.path.append('/repos/smote_msfb/functions')
from filter_anomalous_minority_samples import filter_anomalous_minority_samples

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
       
    class_algo = eval(config["get_minority_obs_to_resample"]["classification_model"])
    
    if config['logging']['diagnostic']:
        print("Classification algo is :",class_algo)
    
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

    ## Code to remove anamoly samples from the minority samples. 
    if config['get_minority_obs_to_resample']['filter_anamoly']:
        ## Step 1: Filter anomalous minority samples from the data
        valid_minority_indices = filter_anomalous_minority_samples(
        x_train, y_train,
        contamination=config["get_minority_obs_to_resample"].get("anomaly_contamination", 0.05),
        random_state=config["random_state"]
        )
    
        if config['logging']['diagnostic']:
            total_minority = np.sum(y_train==1)
            print(f"Total minority samples before anomaly filtering: {total_minority}")
            print(f"Minority samples after anomaly filtering: {len(valid_minority_indices)}")
    
        # Find minority class samples
        minority_indices = valid_minority_indices 
    else:
        minority_indices = np.where(y_train == 1)[0]

    # Identify misclassified minority samples - 
    miscls = [i for i in minority_indices if y_proba_cv[i] <= config["get_minority_obs_to_resample"]["limit_miss_classification_prob"] ]
    
    if len(miscls) == 0:
        if config['logging']['diagnostic']:
            print(f"All minority class samples are getting classified correctly using {config['get_minority_obs_to_resample']['limit_miss_classification_prob']:.2f} probability limit. No applying any lower filter")
        miscls = [i for i in minority_indices if y_proba_cv[i] <= config["get_minority_obs_to_resample"]["limit_miss_classification_prob_revised"] ]

    if config['logging']['diagnostic']:
        print("Final no. of incorrectly classified samples from bagging filter :",len(miscls))    
    
    # Filter by predicted probability
    filtered = [i for i in miscls if y_proba_cv[i] >= config["get_minority_obs_to_resample"]["prob_threshold"] ]

    # Fallback if too few filtered
    if len(filtered) / len(minority_indices) < config["get_minority_obs_to_resample"]["min_fraction"]:
        
        if config['logging']['diagnostic']:
            print(f"Fewer than {config['get_minority_obs_to_resample']['min_fraction']} samples after filtering with threshold {config['get_minority_obs_to_resample']['prob_threshold']}. Not using any noise removal filters.")        
        #filtered = [i for i in miscls if y_proba_cv[i] >= config["get_minority_obs_to_resample"]["fallback_threshold"]]
        filtered = miscls

    # Reporting
    if config['logging']['diagnostic']:
        print(f"Total minority samples: {len(minority_indices)}")
        print(f"Misclassified by bagged classification models: {len(miscls)}")
        print(f"After filtering proba < {config['get_minority_obs_to_resample']['prob_threshold']:.2f}: {len(filtered)} remain")

    # Also return probabilities for filtered samples
    filtered_probs = [y_proba_cv[i] for i in filtered]

    return filtered, filtered_probs
