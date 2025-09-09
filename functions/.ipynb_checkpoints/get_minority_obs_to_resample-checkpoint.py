from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
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
    
    # Get the predicted probabilities for the minority class samples only
    minority_probs = y_proba_cv[minority_indices]
    
    print('=====================================================================')
    print('Total no. of minority samples in the data :', len(minority_indices))

    # Calculate the actual probability values at these percentiles
    lower_thresh = np.percentile(minority_probs, config["get_minority_obs_to_resample"]['hard_classify_lower_limit'])
    upper_thresh = np.percentile(minority_probs, config["get_minority_obs_to_resample"]['hard_classify_upper_limit'])

    # Select indices of minority samples within these percentiles
    percentile_mask = (minority_probs >= lower_thresh) & (minority_probs <= upper_thresh)
    hard_minority_indices = minority_indices[percentile_mask]                    
    
    print('Total no. of hard minority samples identified for resampling :', len(hard_minority_indices))
    print('=====================================================================')
    
    ## If not minority samples are missclassified using 0.8 prob limit, Use the full minority class samples for resampling. 
    ## This negates any gains from resampling the focussed resampling. But we do not have a choice
    if len(hard_minority_indices) == 0:
        print("No minority samples identified as hard to classify.Resetting to the full minority samples")
        hard_minority_indices = minority_indices      
            
    return hard_minority_indices
