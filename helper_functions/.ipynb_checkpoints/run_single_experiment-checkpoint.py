
import numpy as np
import pandas as pd
import pickle
import sys
import gc
import os
import yaml

## Modelling related packages
from sklearn.model_selection import train_test_split
from sklearn import metrics
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
from random import choices
from sklearn.datasets import make_classification
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.model_selection import StratifiedKFold

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

## Classification algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTEN
from sklearn.neighbors import KNeighborsClassifier
#import xgboost as xgb
#from sklearn.svm import SVC

from sklearn.metrics import jaccard_score
#from generate_imbalanced_data import generate_imbalanced_data
#from create_new_sampled_data import create_new_sampled_data
from create_folder_if_not_exists import create_folder_if_not_exists
from train_model import train_model

def run_single_experiment(path, full_data, mode ="new"):      
    """
    This function helps us to run experiments on the publicly available datasets. 
    
    path -> output location where output from this experiment will be stored. 
    mode -> 'new' means the code will generate a new cross validation folds. 'old' will specify to read the folds from the 
            disk to ensure exactly the same folds for experimentation.
    
    """
    n_columns = full_data.shape[1] - 1
    
    ### THESE ARE THE FINAL DATASETS FOR BUILDING THE MODELS.
    x_train = full_data.iloc[:,0:n_columns]
    y_train = full_data['target']
    
    # Define a Jaccard distance function for binary features
    def jaccard_distance(x, y):
        # x and y are 1D arrays
        return 1 - jaccard_score(x, y)

    # Wrap Jaccard distance in a pairwise-compatible format
    def jaccard_distance_pairwise(X, Y):
        from sklearn.metrics import pairwise_distances
        return pairwise_distances(X, Y, metric=lambda x, y: 1 - jaccard_score(x, y))    
    
    if mode == "new":
        print("Generating new spilts for cross validation and storing to desk.")
        ## Do the cross validation spilts outside so that the variance in accuracy metric is purely because of re-sampling algo
        kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        splits = list(kf.split(x_train, y_train))
        
        # Save the splits to a file
        with open(path + "/cv_splits.pkl", "wb") as f:
            pickle.dump(splits, f)        
    else:
        print("Reading existing spilts stored in the folder.")
        # Load the splits from the file
        with open(path + "/cv_splits.pkl", "rb") as f:
            splits = pickle.load(f)
        
    # Dictionary of models
    models = {
    "logistic_regression": LogisticRegression(penalty='l2', random_state=42),    # class_weight='balanced',      
    "naive_bayes": BernoulliNB(),    
    "knn_jaccard": KNeighborsClassifier(n_neighbors=5, metric=jaccard_distance)     
    }    
    
#     models = {
#     "logistic_regression": LogisticRegression(solver='lbfgs', penalty='l2', class_weight='balanced', random_state=42),
#     "naive_bayes": BernoulliNB(),
#     "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
#     "xgboost": xgb.XGBClassifier(objective='binary:logistic',  # binary classification
#                                  eval_metric='auc',        # or use 'auc' for ROC AUC
#                                  use_label_encoder=False,      # needed for newer versions
#                                  random_state=42), 
#     "knn": KNeighborsClassifier(),
#     "svm": SVC(kernel='linear',            # You can try 'rbf' or 'poly' too
#                C=1.0,                      # Regularization strength
#                probability=True,           # Enable probability estimates (for AUC)
#                class_weight='balanced',    # Very important for imbalanced data
#                random_state=42 ),
#     "knn_jaccard": KNeighborsClassifier(n_neighbors=5, metric=jaccard_distance)
#     }
    
    ### The code expects a config yaml file for smote_msfb to kept in the smote_msfb_config folder in the ouput folder
    with open("/repos/smote_msfb/functions/config.yaml", "r") as f:
        smote_msfb_config1 = yaml.safe_load(f)
    
    resample_algo = {    
    "smote_msfb1": smote_msfb_config1,
    "smoten": SMOTEN(random_state=42, k_neighbors = 5), # sampling_strategy = desired_imbal_ratio                
    "no_sampling": "no_sampling",
    "random_oversample": RandomOverSampler(random_state=42)    
    }
    
    for resample_name, resample_model in resample_algo.items():
        # Print models to verify
        for name, model in models.items():

            print(f"{name}: {model}")
            print(f"{resample_name}: {resample_model}")
            version = resample_name+"_"+name                   
            train_model(x_train, y_train, model, resample_model, path, version, splits)
        
    return 1

