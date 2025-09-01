import numpy as np
import pandas as pd

from get_top_relevant_feature_list import get_top_relevant_feature_list
from get_knn_minority_train_samples import get_knn_minority_train_samples
#from visualize_high_dim_data import visualize_high_dim_data
from create_synthetic_obs import create_synthetic_obs

def create_new_sampled_data(y_train, x_train, n_rows, imbalance_ratio, n_columns, n_cols_imp_var, no_of_nn):
    
    df = pd.concat([x_train, y_train], axis=1, ignore_index=True)
    df.reset_index(inplace=True)
    df.drop('index', axis=1, inplace=True)
    
    #print("column names of the joint dataset df BEFORE RENAME :", df.columns)
    
    df.rename(columns={df.columns[-1]: 'target'}, inplace=True)
    #print("column names of the joint dataset df AFTER RENAME :", df.columns)
    
    #print("shape of the joint dataset df :", df.shape)
    #print("column names of the joint dataset df :", df.columns)
    
    ## Get top features for generating new synthetic data
    feature_importance = get_top_relevant_feature_list(df.iloc[:,0:n_columns], df['target'])
    
    ## Filter the training dataset to keep only the n_cols_imp_var relevant features in the dataset
    df_imp = df.iloc[ : ,feature_importance['col_no'].iloc[0:n_cols_imp_var] ] 
    
    ## Get nearest neighbours for each of the minority class observation
    distances, indices = get_knn_minority_train_samples(df_imp,df['target'], no_of_nn )
    
    ## Code to remove all majority observations other than the ones in 10 nearest neighbours of minority class observations
    filtered_index = np.unique(indices)
    print("Total no. of unqiue observations present in the NN of object:", len(filtered_index))
    print("Total no. of observations in the input dataset :", df_imp.shape[0])
    print(f" % of distinct observations in KNN of the dataset {df_imp.shape[0]} is : {len(filtered_index) / df_imp.shape[0]:.2%}")
    print("This metric quantifies hubness phenomena in high dimensional data.")
    print("Lower the percentage above, Higher is the hubness phenomena.")

        
    # Create an initial empty DataFrame with ~3500 columns
    synthetic_obs = pd.DataFrame(columns= df.drop('target', axis=1).columns)
    synthetic_obs['idx'] = 0
    synthetic_obs['neighbor_idx'] = 0


    # Loop through each observation in y_train and process only minority class observations
    for idx in range(len(df['target'])): #range(0,20): 
        
        if (idx % 2000) == 0:
            print("Processed records :", idx)
        
    #print("idx :", idx)
        if df['target'][idx] == 1:  # Only for minority class observations
            # Get the minority class observation's features using .iloc for pandas DataFrame
            minority_features = df.drop('target', axis=1).iloc[idx].values  # .values converts the row to a numpy array
        
            # Get the indices of the 10 nearest neighbors for this observation
            nearest_neighbors_indices = indices[idx]
        
            # Loop through each of the nearest neighbors
            for neighbor_idx in nearest_neighbors_indices:
                
                if (neighbor_idx == idx): 
                    #print(f"Both the neighbour idx {neighbor_idx} and obs idx {idx} are the same")
                    continue
                
                if (df['target'][neighbor_idx] == 0):  ## We are ONLY CREATING SYNTHETIC OBSERVATIONS FOR MINORITY - MINORITY CLASS OBS
                    continue
                
                # Get the neighbor's features using .iloc for pandas DataFrame
                neighbor_features = df.drop('target', axis=1).iloc[neighbor_idx].values  # .values converts the row to a numpy array
            
                # Call the function to print the two observations
                new_synthetic_obs = create_synthetic_obs(minority_features, neighbor_features, df['target'][neighbor_idx], np.array(feature_importance['col_no'].iloc[0:n_cols_imp_var]) , 0.8)
                #print("Shape of new_synthetic_obs :", new_synthetic_obs.shape )
                #print("new_synthetic_obs :", new_synthetic_obs )
                new_synthetic_obs_pd =  pd.DataFrame([new_synthetic_obs]) 
                new_synthetic_obs_pd.columns = df.drop('target', axis=1).columns
                new_synthetic_obs_pd['idx'] = idx
                new_synthetic_obs_pd['neighbor_idx'] = neighbor_idx
            
                synthetic_obs = pd.concat([synthetic_obs, new_synthetic_obs_pd ], ignore_index=True)

    data_to_be_added = synthetic_obs.iloc[:,0:n_columns]
    # data_to_be_added['target'] = 1 ## We have generated all minority class observations
    
    data_to_be_added['target'] = 2 ## Third label for visualization
    print("Shape of the data to be added :", data_to_be_added.shape)
    print("For visualization, we are adding the additional data with target = 2 label. ")
         
    data_to_be_added.columns.values[0:n_columns] = x_train.columns
    
    #print("shape of the dataset to be added :", data_to_be_added.shape)
    #print("shape of the dataset x_train :", x_train.shape)
    
    #print("column names of the dataset to be added df to y_train :", data_to_be_added['target'].columns)
    #print("****************************************************************")
    #print("column names of the dataset y_train :", y_train.columns)
    
    x_train_upd = pd.concat([x_train, data_to_be_added.iloc[:,0:n_columns] ], axis = 0, ignore_index=True)
    y_train_upd = pd.concat([y_train, pd.DataFrame(data_to_be_added['target']) ], axis = 0, ignore_index=True)
    
    #print("shape of the dataset x_train_upd :", x_train_upd.shape)
    
    return x_train_upd, y_train_upd