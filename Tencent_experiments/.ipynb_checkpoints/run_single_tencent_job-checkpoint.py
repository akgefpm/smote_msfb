
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix
import pickle
import sys
import gc
import os

import snappy
import fastparquet

import zipfile
import pandas as pd
from io import StringIO

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import jaccard_score
import numpy as np

from sklearn.preprocessing import LabelEncoder
import time

from expand_cols_28Jan import expand_cols_28Jan

sys.path.append('/repos/smote_msfb/helper_functions/')  # Adding the 'src' directory to sys.path

from sklearn.model_selection import StratifiedKFold
from create_folder_if_not_exists import create_folder_if_not_exists
from train_model import train_model

from imblearn.over_sampling import SMOTEN
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from create_folder_if_not_exists import create_folder_if_not_exists

def run_single_tencent_job(aid_sel, path, minority_n = 400, majority_n = 2000 ):
    
    ### Data processing prior to this is present in the file named - "Tencent_Data_Prep_28Jan2025.ipynb"
    ### Read the Parquet file into a pandas DataFrame
    train_data_pre_ftr = pd.read_parquet('/domino/datasets/local/smote_msfb/tencent_data/Train_data_pre_feature_engg.snap.parquet')
    
    #path = path + str(aid_sel)+"/"
    # Example usage
    print("Shape of the training dataset before one hot encoding the dataset :", train_data_pre_ftr.shape)
    
    print("train_data_pre_ftr.head(3) :" , train_data_pre_ftr.head(3))
    
    train_data_pre_ftr = train_data_pre_ftr[ train_data_pre_ftr['aid'] == aid_sel ]
    
    print(" Balance of response variable after subsetting for advertiser ID :", train_data_pre_ftr[['uid','label']].groupby(['label']).count())
    
    # Separate the classes
    minority_class = train_data_pre_ftr[train_data_pre_ftr['label'] == 1]
    majority_class = train_data_pre_ftr[train_data_pre_ftr['label'] == 0]

    # Sample from each class
    
    replace = len(minority_class) < minority_n

    minority_sample = minority_class.sample(n=minority_n, random_state=42, replace=replace)
    #minority_sample = minority_class.sample(n=minority_n, random_state=42)
    majority_sample = majority_class.sample(n=majority_n, random_state=42)

    # Combine them into one DataFrame
    sampled_df = pd.concat([minority_sample, majority_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Check class distribution (optional)
    print(" Balance of response variable after sampling :", sampled_df['label'].value_counts())
    
    ### These are columns where there is only one value per cell and hence these can be directly one-hot-encoded

    categorical_cols = ['creativeSize','adCategoryId','productType','age','gender','education','consumptionAbility','LBS','carrier','house']

    # Convert selected int64 columns to object (if not already)
    sampled_df[categorical_cols] = sampled_df[categorical_cols].astype(str)

    # One-hot encode with consistent naming format: column_value
    sampled_df = pd.get_dummies(sampled_df, columns=categorical_cols, prefix=categorical_cols, prefix_sep='_')

    ## These are columns where "--- each cell may contain more than one value separated by spaces ---"
    col_list = ['ct','os','interest1','interest2','interest3','interest4','interest5','appIdAction',
                'appIdInstall','kw1','kw2','kw3','topic1','topic2','topic3','marriageStatus']

    for col_name in col_list:
        #print("Column being transformed :", col_name)
        #print("Shape of the dataset before one hot encoding :", sampled_df.shape)
        sampled_df = expand_cols_28Jan( sampled_df , col_name)
        #print("Shape of the dataset after one hot encoding :", sampled_df.shape)
    
    print("sampled_df.head(3)   :",  sampled_df.head(3) )
    
    sampled_df = sampled_df.astype(bool).astype(int)
    
    ## Check the unique values in the co-variate space - It should be 0 OR 1. 
    print(" Unique values in the covariate space : ",pd.unique(sampled_df[sampled_df.columns[3:]].iloc[0:2].values.ravel()) )
    
    y_train = sampled_df['label']
    y_train.name = 'target'
    x_train = sampled_df.iloc[:, 3:]
    
    print("Shape of x_train :", x_train.shape)
    print("Shape of y_train :", y_train.shape)
    
    print("========================== Modelling Part begins ===================================")
    
    # Make sure it's a Series (which it is)
    class_counts = y_train.value_counts()

    # Get counts
    majority_class_obs_num = class_counts.max()
    minority_class_obs_num = class_counts.min()

    # Calculate imbalance ratio
    obs_imb_ratio = minority_class_obs_num / (minority_class_obs_num + majority_class_obs_num)

    print(f"Imbalance ratio of dataset is: {obs_imb_ratio:.4f}")
    
    n_rows = x_train.shape[0]
    imbalance_ratio = obs_imb_ratio #  0.25  # 10% minority class, 90% majority class
    n_columns = x_train.shape[1] 
    n_cols_imp_var = int(n_columns*0.1)  ## no. of top important variables to be chosen
    no_of_nn = int ( ( n_rows - (2*imbalance_ratio * n_rows) ) / ( imbalance_ratio * n_rows) ) 
    
    print("No. of nearest neighbours for each minority class observation :",no_of_nn )
    
    # Remove columns with only a single unique value
    x_train = x_train.loc[:, x_train.nunique() > 1]
    
    # Get columns in x_train that have only one unique value (e.g., only 0 or only 1) - These need to be removed
    constant_cols = [col for col in x_train.columns if x_train[col].nunique() == 1]
    print(" Columns with same values in all rows of data ",constant_cols)
    
    print("Shape of x_train :", x_train.shape)
    print("Shape of y_train :", y_train.shape)
    print("No. of observations :", y_train.value_counts() )

    ## Do the cross validation spilts outside so that the variance in accuracy metric is purely because of re-sampling algo
    kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    splits = list(kf.split(x_train, y_train))
    
    cv_split_path = os.path.join(path +"/", "cv_splits.pkl")

    # Check if CV splits file already exists
    if os.path.exists(cv_split_path):
        print("Reading existing splits stored in the folder.")
        with open(cv_split_path, "rb") as f:
            splits = pickle.load(f)
    else:
        print("Generating new splits for cross-validation and storing to disk.")
        kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        splits = list(kf.split(x_train, y_train))

        # Save the splits to a file
        with open(cv_split_path, "wb") as f:
            pickle.dump(splits, f)

    # Dictionary of models
    models = {
    "logistic_regression": LogisticRegression(solver='lbfgs', penalty='l2', random_state=42),
    "naive_bayes": BernoulliNB(),
    # "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    # "xgboost": xgb.XGBClassifier(objective='binary:logistic',  # binary classification
    #                                  eval_metric='auc',        # or use 'auc' for ROC AUC
    #                                  use_label_encoder=False,      # needed for newer versions
    #                                  random_state=42), 
    #"knn": KNeighborsClassifier(),
    #"knn_jaccard": KNeighborsClassifier(n_neighbors=5, metric=jaccard_distance),
    "svm": SVC(kernel='linear',            # You can try 'rbf' or 'poly' too
               C=1.0,                      # Regularization strength
               probability=True,           # Enable probability estimates (for AUC)
              # class_weight='balanced',    # Very important for imbalanced data
               random_state=42 )
    }

    resample_algo = {
    "minority_focussed_smotten_weighted_jaccard": "minority_focussed_smotten_weighted_jaccard",
    #"minority_focussed_smotten": "minority_focussed_smotten",    
    "smoten": SMOTEN(random_state=42, k_neighbors = 5) #, # sampling_strategy = desired_imbal_ratio
    # "no_sampling": "no_sampling",
    # "random_oversample": RandomOverSampler(random_state=42)    
    }

    for resample_name, resample_model in resample_algo.items():
    # Print models to verify
        for name, model in models.items():

            print(f"{name}: {model}")
            print(f"{resample_name}: {resample_model}")
            version = resample_name+"_"+name
            train_model(x_train, y_train, model, resample_model, path, version, splits)
    
    
    
    