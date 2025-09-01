
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc
from imblearn.under_sampling import RandomUnderSampler

import pickle
import sys
import gc
import os
import time

from manual_roc_points import manual_roc_points
from visualize_resample import visualize_resample
from smoten_select_feature import smoten_select_feature
from minority_focussed_smotten import minority_focussed_smotten

def train_model(X_scaled, y, model, resample_model, path, version, splits):
    
    #print(" ")
    #print("Shape of X_scaled :", X_scaled.shape)
    #print("Shape of y :", y.shape)
    
    # Reset the indices of X_scaled and y (just in case there are any index issues)
    X_scaled = X_scaled.to_numpy() 
    y = y.to_numpy() 
    
    ## Do the cross validation spilts outside so that the variance in accuracy metric is purely because of re-sampling algo
    #kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
       
    # Lists to store metrics across the folds
    test_precision = []
    test_recall = []
    
    test_roc_auc = []
    roc_data_list = [] 
    roc_data_list_m = []
    resampling_time = []
    
    fold_num = 0

    # Perform 10-fold cross-validation
    for train_index, test_index in splits: # kf.split(X_scaled, y):
        
        #print("Train_Index :", sum(train_index) )
        #print("Test_Index :", sum(test_index) )
        
        # Split data into train and test based on the current fold
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Only output the files if the first fold is 
        pd.DataFrame(X_test).to_csv(path + "/X_test_"+str(fold_num) + ".zip", index=False, compression="zip")
        pd.DataFrame(X_train).to_csv(path + "/X_train_"+str(fold_num) + ".zip", index=False, compression="zip")
        pd.DataFrame(y_test).to_csv(path + "/y_test_"+str(fold_num) + ".zip", index=False, compression="zip")
        pd.DataFrame(y_train).to_csv(path + "/y_train_"+str(fold_num) + ".zip", index=False, compression="zip")
                        
        # Count occurrences of 0s and 1s
        #print(f"y_train Count of 0s: {np.bincount(y_train)[0]} || Count of 1s: {np.bincount(y_train)[1]}")
                       
        #print("Shape of X_train :", X_train.shape)
        #print("Shape of y_train :", y_train.shape)
        #print("  ")
        #print("Shape of X_test :", X_test.shape)
        #print("Shape of y_test :", y_test.shape)           
        
        ## Apply your resampling algo here - ONLY ON TRAIN DATA - THIS WILL ENSURE NO DATA LEAKAGE
        if resample_model == "smotten_sf_mfs_p":
            
            ## File path where the file should be stored
            X_train_file_path = path + "/X_train_upd_"+str(resample_model)+"_"+str(fold_num)+"_resampled_data.zip"
            y_train_file_path = path + "/y_train_upd_"+str(resample_model)+"_"+str(fold_num)+"_resampled_data.zip"
            
            if ( os.path.exists(X_train_file_path) & os.path.exists(y_train_file_path) ):
            ## Case 1 - if the resampled file already exists in the folder. then just read that file
                print(f"File found: Resampled x_train_upd for fold {fold_num} for {resample_model} {X_train_file_path}")
                
                X_train_upd_pd = pd.read_csv(X_train_file_path, index_col=0)
                y_train_upd_pd = pd.read_csv(y_train_file_path, index_col=0)
                
                X_train_upd = np.array(X_train_upd_pd)
                y_train_upd = np.array(y_train_upd_pd).ravel()
                
                print("Shape of the file read from disk :", X_train_upd.shape)
                
            else:
            ## Case 2 - if the resampled file does not exists in the folder. Do the resampling and store the file
                print("This should come only once ----> performing resampling via "+str(resample_model)+" and storing the file "+str(fold_num))
                
                ## Resampling code
                sampling_strategy = {1: (np.bincount(y_train)[0]  - np.bincount(y_train)[1]) }
                print("sampling strategy :", sampling_strategy)
                
                start = time.perf_counter()
                
                X_train_upd , y_train_upd = smoten_select_feature(X_train, y_train, sampling_strategy, top_feature_percent=20, k_neighbors=6, random_state=42)
                
                end = time.perf_counter()
                print(f"Resampling took {resample_model} : {end - start:.3f} seconds")
                
                resampling_time.append([ str(resample_model), str(fold_num), (end - start) ])
                
                print("Shape of the file written to disk :", X_train_upd.shape)
                pd.DataFrame(X_train_upd).to_csv(path + "/X_train_upd_"+str(resample_model)+"_"+str(fold_num)+"_resampled_data.zip", index=False, compression="zip")
                pd.DataFrame(y_train_upd).to_csv(path + "/y_train_upd_"+str(resample_model)+"_"+str(fold_num)+"_resampled_data.zip", index=False, compression="zip")
            
            #X_train_upd , y_train_upd = create_smoten_sampled_data(y_train, X_train, n_columns, n_cols_imp_var, desired_imbal_ratio)
        
        elif resample_model == "minority_focussed_smotten":
            
            ## File path where the file should be stored
            X_train_file_path = path + "/X_train_upd_"+str(resample_model)+"_"+str(fold_num)+"_resampled_data.zip"
            y_train_file_path = path + "/y_train_upd_"+str(resample_model)+"_"+str(fold_num)+"_resampled_data.zip"
            
            if ( os.path.exists(X_train_file_path) & os.path.exists(y_train_file_path) ):
            ## Case 1 - if the resampled file already exists in the folder. then just read that file
                print(f"File found: Resampled x_train_upd for fold {fold_num} for {resample_model} {X_train_file_path}")
                
                X_train_upd_pd = pd.read_csv(X_train_file_path, index_col=0)
                y_train_upd_pd = pd.read_csv(y_train_file_path, index_col=0)
                
                X_train_upd = np.array(X_train_upd_pd)
                y_train_upd = np.array(y_train_upd_pd).ravel()
                
                print("Shape of the file read from disk :", X_train_upd.shape)
                
            else:
            ## Case 2 - if the resampled file does not exists in the folder. Do the resampling and store the file
                print("This should come only once ----> performing resampling via "+str(resample_model)+" and storing the file "+str(fold_num))
                
                ## Resampling code
                sampling_strategy = (np.bincount(y_train)[0]  - np.bincount(y_train)[1])
                print("sampling strategy :", sampling_strategy)
                
                start = time.perf_counter()
                
                X_train_upd , y_train_upd = minority_focussed_smotten(X_train, y_train, 
                                                                      no_of_synthetic_samples_to_be_generated = sampling_strategy, 
                                                                      neighbours_to_consider_for_neighbourhood = 8)
                
                end = time.perf_counter()
                print(f"Resampling took {resample_model} : {end - start:.3f} seconds")
                
                resampling_time.append([ str(resample_model), str(fold_num), (end - start) ])
                
                print("Shape of the file written to disk :", X_train_upd.shape)
                pd.DataFrame(X_train_upd).to_csv(path + "/X_train_upd_"+str(resample_model)+"_"+str(fold_num)+"_resampled_data.zip", index=False, compression="zip")
                pd.DataFrame(y_train_upd).to_csv(path + "/y_train_upd_"+str(resample_model)+"_"+str(fold_num)+"_resampled_data.zip", index=False, compression="zip")
        
        elif resample_model == "minority_focussed_smotten_weighted_jaccard":
            
            ## File path where the file should be stored
            X_train_file_path = path + "/X_train_upd_"+str(resample_model)+"_"+str(fold_num)+"_resampled_data.zip"
            y_train_file_path = path + "/y_train_upd_"+str(resample_model)+"_"+str(fold_num)+"_resampled_data.zip"
            
            if ( os.path.exists(X_train_file_path) & os.path.exists(y_train_file_path) ):
            ## Case 1 - if the resampled file already exists in the folder. then just read that file
                print(f"File found: Resampled x_train_upd for fold {fold_num} for {resample_model} {X_train_file_path}")
                
                X_train_upd_pd = pd.read_csv(X_train_file_path, index_col=0)
                y_train_upd_pd = pd.read_csv(y_train_file_path, index_col=0)
                
                X_train_upd = np.array(X_train_upd_pd)
                y_train_upd = np.array(y_train_upd_pd).ravel()
                
                print("Shape of the file read from disk :", X_train_upd.shape)
                
            else:
            ## Case 2 - if the resampled file does not exists in the folder. Do the resampling and store the file
                print("This should come only once ----> performing resampling via "+str(resample_model)+" and storing the file "+str(fold_num))
                
                ## Resampling code
                sampling_strategy = (np.bincount(y_train)[0]  - np.bincount(y_train)[1])
                print("sampling strategy :", sampling_strategy)
                
                # Print class counts before resampling
                print("Before resampling - label 0 (majority):", (y_train == 0).sum(),
                          ", label 1 (minority):", (y_train == 1).sum())
                
                start = time.perf_counter()
                
                X_train_upd , y_train_upd = minority_focussed_smotten(X_train, y_train, 
                                                                      no_of_synthetic_samples_to_be_generated = sampling_strategy, 
                                                                      neighbours_to_consider_for_neighbourhood = 5)
                
                end = time.perf_counter()
                
                # Print class counts after resampling
                print("After resampling - label 0 (majority):", (y_train_upd == 0).sum(),
                      ", label 1 (minority):", (y_train_upd == 1).sum())
                
                print(f"Resampling took {resample_model} : {end - start:.3f} seconds")
                
                resampling_time.append([ str(resample_model), str(fold_num), (end - start) ])
                
                print("Shape of the file written to disk :", X_train_upd.shape)
                pd.DataFrame(X_train_upd).to_csv(path + "/X_train_upd_"+str(resample_model)+"_"+str(fold_num)+"_resampled_data.zip", index=False, compression="zip")
                pd.DataFrame(y_train_upd).to_csv(path + "/y_train_upd_"+str(resample_model)+"_"+str(fold_num)+"_resampled_data.zip", index=False, compression="zip")
        
        
        elif resample_model == "no_sampling":
            
            ## There is no need to store this data            
            X_train_upd = X_train
            y_train_upd = y_train               
            
        else:            
            ## File path where the file should be stored
            X_train_file_path = path + "/X_train_upd_"+str(resample_model)+"_"+str(fold_num)+"_resampled_data.zip"
            y_train_file_path = path + "/y_train_upd_"+str(resample_model)+"_"+str(fold_num)+"_resampled_data.zip"
            
            if ( os.path.exists(X_train_file_path) & os.path.exists(y_train_file_path) ):
            ## Case 1 - if the resampled file already exists in the folder. then just read that file
                print(f"File found: Resampled x_train_upd for fold {fold_num} for {resample_model} {X_train_file_path}")
                X_train_upd_pd = pd.read_csv(X_train_file_path, index_col=0)
                y_train_upd_pd = pd.read_csv(y_train_file_path, index_col=0)
                
                X_train_upd = np.array(X_train_upd_pd)
                y_train_upd = np.array(y_train_upd_pd).ravel()
                
                print("Shape of the file read from disk :", X_train_upd.shape)                
            else:
            ## Case 2 - if the resampled file does not exists in the folder. Do the resampling and store the file
                print("This should come only once ----> performing resampling via "+str(resample_model)+" and storing the file "+str(fold_num))
                
                # Print class counts before resampling
                print("Using ",str(resample_model)," Before resampling - label 0 (majority):", (y_train == 0).sum(),
                          ", label 1 (minority):", (y_train == 1).sum())
                
                start = time.perf_counter()
                
                ## Resampling code
                X_train_upd , y_train_upd = resample_model.fit_resample(X_train, y_train)
                
                end = time.perf_counter()
                print(f"Resampling took {resample_model} : {end - start:.3f} seconds")
                
                # Print class counts after resampling
                print("Using ",str(resample_model)," After resampling - label 0 (majority):", (y_train_upd == 0).sum(),
                      ", label 1 (minority):", (y_train_upd == 1).sum())
                
                resampling_time.append([ str(resample_model), str(fold_num), (end - start) ])
                
                print("Shape of the file written to disk :", X_train_upd.shape)
                pd.DataFrame(X_train_upd).to_csv(path + "/X_train_upd_"+str(resample_model)+"_"+str(fold_num)+"_resampled_data.zip", index=False, compression="zip")
                pd.DataFrame(y_train_upd).to_csv(path + "/y_train_upd_"+str(resample_model)+"_"+str(fold_num)+"_resampled_data.zip", index=False, compression="zip")
            
            
        print(f"y_train_upd Count of 0s: {np.bincount(y_train_upd.astype(int))[0]} || Count of 1s: {np.bincount(y_train_upd.astype(int))[1]}")
        
        #print("Shape of X_train_upd :", X_train_upd.shape)
        #print("Shape of y_train_upd :", y_train_upd.shape)
        
        # Fit the model - ON THE RESAMPLED DATA
        model.fit(X_train_upd, y_train_upd)
        
        fold_num = fold_num + 1
        
        # Save the splits to a file
        with open(path + "/Model_"+str(version)+"_"+str(fold_num)+".pkl", "wb") as f:
            pickle.dump(splits, f)  
                
        # Predict on the test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC-AUC
        
        ## Put the function to add the visualization plot
        #visualize_resample(X_train, y_train, X_train_upd, y_train_upd, path, version)
        
        #pd.DataFrame(y_test).to_csv(path + "/y_test_"+str(version)+"_"+str(fold_num)+".csv")
        #pd.DataFrame(y_pred).to_csv(path + "/y_pred_"+str(version)+"_"+str(fold_num)+".csv")
        
        # This file has probabilities of each observation
        pd.DataFrame(y_prob).to_csv(path + "/y_prob_"+str(version)+"_"+str(fold_num)+".zip", index=False, compression="zip")
        
        # Compute metrics for this fold
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        
        common_thresholds = np.linspace(0, 1, 101)  # 0.00 to 1.00 in steps of 0.01
        fpr_m, tpr_m = manual_roc_points(y_test, y_prob, common_thresholds)
        
        # Create a DataFrame for this fold
        roc_df = pd.DataFrame({
            'Fold': fold_num,
            'FPR': fpr,
            'TPR': tpr,
            'Threshold': thresholds
        })

        # Create a DataFrame for this fold
        roc_df_m = pd.DataFrame({
            'Fold': fold_num,
            'FPR': fpr_m,
            'TPR': tpr_m,
            'Threshold': common_thresholds
        })
        
        roc_data_list.append(roc_df)  
        roc_data_list_m.append(roc_df_m) 
        
        roc_auc = auc(fpr, tpr)
        
        #print("test_precision for ",version," ",test_precision)
        #print("test_recall for", version, " ", test_recall)
        #print("test_roc_auc for", version, " ", roc_auc)
        
        # Append the metrics to their respective lists
        test_precision.append(precision)
        test_recall.append(recall)  
        test_roc_auc.append(roc_auc)

    # Calculate average metrics across all folds
    avg_precision = np.mean(test_precision)
    avg_recall = np.mean(test_recall)
    avg_roc_auc = np.mean(test_roc_auc)

    # Print the metrics
    print(version," - Average Precision across 10 folds:", avg_precision)
    print(version," - Average Recall across 10 folds:", avg_recall)
    print(version," - Average ROC-AUC across 10 folds:", avg_roc_auc)    
    
    roc_data_all_folds = pd.concat(roc_data_list, ignore_index=True)
    roc_data_all_folds_m = pd.concat(roc_data_list_m, ignore_index=True)
        
    # Create a DataFrame from the lists
    metrics_df = pd.DataFrame({
        'Precision': test_precision,
        'Recall': test_recall,
        'ROC_AUC': test_roc_auc,      
    })
    
    cv_metrics_df = pd.DataFrame({
        'Avg_Precision': [avg_precision],
        'Avg_Recall': [avg_recall],
        'Avg_ROC_AUC': [avg_roc_auc]
    })
    
    resampling_time_df = pd.DataFrame(resampling_time, columns=["resample_model", "fold_num", "resample_time"])
    
    # Save to file
    roc_data_all_folds_m.to_csv(path + "/roc_data_manual_threshold_" + str(version) + ".zip", index=False, compression="zip")
    roc_data_all_folds.to_csv(path + "/roc_data_" + str(version) + ".zip", index=False, compression="zip")
    metrics_df.to_csv(path + "/metrics_"+version+".zip", index=False, compression="zip")
    cv_metrics_df.to_csv(path + "/Cross_Validation_metrics_"+version+".zip", index=False, compression="zip")
    resampling_time_df.to_csv(path + "/Resampling_time_metrics_"+version+".zip", index=False, compression="zip")
        
    return 1 
