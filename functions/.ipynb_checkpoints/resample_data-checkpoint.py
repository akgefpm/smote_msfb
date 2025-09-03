import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.feature_selection import mutual_info_classif    
import matplotlib.pyplot as plt
import math
import sys

sys.path.append('/repos/smote_msfb/functions')
from weighted_jaccard_distance import weighted_jaccard_distance
from get_hubness_score_minority_class import get_hubness_score_minority_class


def resample_data(x_train, y_train, minority_resample_list, config):
    """
    
    Generates synthetic samples for the given minority_resample_list based on 5-NN using weighted Jaccard distance.
    To improve robustness in high-dimensional binary data, feature selection using mutual information is performed
    for neighbor search only.
    """   
    
    sampling_strategy = config['main_section']['sampling_strategy']    
    n_neighbors= config['main_section']['neighbours_to_consider_for_neighbourhood']
    perc_MI = config['feature_selection']['Percent_MI']
        
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    # Count majority and minority
    counts = np.bincount(y_train)
    n_majority = counts[0]
    n_minority = counts[1]
    
    target_minority = int(sampling_strategy * n_majority)
    no_of_synthetic_samples_to_be_generated = max(0, target_minority - n_minority)
    
    if config['logging']['diagnostic']:
        print("Based on user input the no. of minority samples to be generated :", no_of_synthetic_samples_to_be_generated)
            
    minority_indices = np.where(y_train == 1)[0]
    x_minority = x_train[minority_indices] 
    
    #######################################################################################################################
    ### Feature selection part of the code :
    ### Part 1 -> The below part selects top features from the data which contribute to 95% of the Mutual Information.
    #######################################################################################################################

    # Compute MI scores
    mi_scores = mutual_info_classif(x_train, y_train, discrete_features=True)

    # Sort MI scores in descending order and get the sorted indices
    sorted_indices = np.argsort(mi_scores)[::-1]
    sorted_mi_scores = mi_scores[sorted_indices]

    # Compute cumulative MI and normalize to get cumulative percentages
    cumulative_mi = np.cumsum(sorted_mi_scores)
    total_mi = cumulative_mi[-1]
    cumulative_perc = cumulative_mi / total_mi

    # Find how many top features are needed to reach perc_MI coverage
    num_features_to_select = np.searchsorted(cumulative_perc, perc_MI) + 1

    # Get the indices of the selected top features
    top_features = sorted_indices[:num_features_to_select]
    
    num_irrelevant_features_removed = len(mi_scores) - len(top_features)

    if config['logging']['diagnostic']:
        print(f"Selected {num_features_to_select} features that account for {perc_MI*100:.1f}% of total mutual information.")
        print(f"No. of irrelevant variables removed for neighbourhood definition: {num_irrelevant_features_removed}")

    #######################################################################################################################
    ### Part 2 -> Create Minority-Minority Neighbourhood based on the reduced no. of features selected based on MI
    #######################################################################################################################    
    
    # Use only top features for neighbor computation
    x_minority_selected = x_minority[:, top_features]
    
    if config['logging']['diagnostic']:
        print("Shape of x_minority_selected :", x_minority_selected.shape )
    
    # Use MI scores of selected features as weights
    weights = mi_scores[top_features]
           
    # Normalize the weights to sum to 1
    weights = weights / np.sum(weights)
    
    # Compute weighted Jaccard distance for all minority to minority data points
    jaccard_dist = weighted_jaccard_distance(x_minority_selected, weights)
    
    if config['logging']['diagnostic']:
        print("weights.shape :", weights.shape)
    
    # Fit KNN on minority samples (on selected features)
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='precomputed')  # +1 to exclude self
    knn.fit(jaccard_dist)    
    
    if config['logging']['diagnostic']:
        print("Shape of jaccard_dist matrix :", jaccard_dist.shape ) ## This is equal to minority * minority matrix
    
    # For each minority sample, find its neighbors from the minority samples only
    neighbor_indices = knn.kneighbors(jaccard_dist, return_distance=False)

    # Exclude self-neighbors (first neighbor is the point itself because distance = 0)
    neighbor_indices = neighbor_indices[:, 1:]
    
    #######################################################################################################################
    ### Part 3 -> Finalize the no. of samples to be generated. Output is minority resample list
    #######################################################################################################################
    
    # Resampling strategy
    if config['logging']['diagnostic']:
        print("No. of synthetic samples to be generated:", no_of_synthetic_samples_to_be_generated)

    if no_of_synthetic_samples_to_be_generated > len(minority_resample_list):
        if config['logging']['diagnostic']:
            print("---- RESAMPLE DATA - NO. OF SYNTHETIC SAMPLES TO BE GENERATED IS HIGHER THAN MISSCLASSIFIED OBSERVATIONS ----")

    if config['logging']['diagnostic']:
        print(f"We are drawing {no_of_synthetic_samples_to_be_generated} samples from an array of size {len(minority_resample_list)}.")
    
    #######################################################################################################################
    ### Part 4 -> Determine the no. of samples to be generated for each identified 
    ## 
    ## Sample evenly across the minority samples identified - While choosing from the miss-classified minority samples, 
    ## a larger set of minority samples for creating synthetic observations. We will first create synthetic samples
    ## from all miss-classified samples evenly as much as possible and then only for the remainder we will use randomness.    
    #######################################################################################################################

    ### Incorporate the hubness metric for each observation here <---- the weight of the metric 
    
    if config['hubness']['incorporate_hubness']:
        if config['logging']['diagnostic']:
            print("Incorporating Hubness score in the definition")
        
        ## Create Hubness score only on the basis of the top_features MI identified above
        if config['hubness']['top_features_only']:
            
            # Use only top features for neighbor computation
            x_train_top_feature = x_train[:, top_features]
            
            ### Incorporate the hubness metric for each observation here <---- the weight of the metric 
            x_minority_hubness_score = get_hubness_score_minority_class(x_train_top_feature, y_train, weights, k=5 )
            
            if config['logging']['diagnostic']:
                print("Shape of the x_minority_hubness_score dict :", x_minority_hubness_score )
                print("Shape of the weights metric :", weights.shape)
                print("Selected minority resamples list by the bagged filter :", minority_resample_list)            
                    
            # Extract hubness scores for candidates
            scores = np.array([x_minority_hubness_score[idx] for idx in minority_resample_list])

            # Avoid division by zero (if any hubness score is 0)
            inv_scores = 1.0 / (scores + 1e-10)

            # Normalize to form probabilities
            probs = inv_scores / inv_scores.sum()

            # Sample without replacement based on probabilities
            minority_resample_list_upd = np.random.choice(
                minority_resample_list,
                size=no_of_synthetic_samples_to_be_generated,
                replace=True,  # allow multiple synthetic samples from same observation
                p=probs
            )
            
            if config['logging']['diagnostic']:
                print("Selected samples for resampling (after adjusting for hubness probability) :,", minority_resample_list_upd)
            
        ## Create hubness score on the basis of full feature space. 
        else:
            
            print("CODE IS NOT WRITTEN HERE. THIS SHOULD NOT COME. ")
        
        
    else:
        if config['logging']['diagnostic']:
            print("Hubness score are NOT incorporated in the resampling.")
    
        minority_resample_list = np.array(minority_resample_list)
        n_sources = len(minority_resample_list)
        total_needed = int(no_of_synthetic_samples_to_be_generated)

        # 1. Base count per source
        base_count = total_needed // n_sources

        # 2. Remaining samples after even allocation
        remainder = total_needed % n_sources

        # 3. Step 1: evenly replicate each sample `base_count` times
        even_part = np.repeat(minority_resample_list, base_count)

        # 4. Step 2: randomly choose `remainder` samples from the list
        random_part = np.random.choice(minority_resample_list, size=remainder, replace=True) 

        # 5. Combine both parts
        minority_resample_list_upd = np.concatenate([even_part, random_part])     

    #######################################################################################################################
    ### Part 5 -> Generate synthetic samples
    #######################################################################################################################
        
    synthetic_samples = []

    for idx in minority_resample_list_upd:
        # Get index of this sample in the minority array

        if config['logging']['diagnostic']:
            print("Minority Sample for which synthetic obs are generated :", idx)

        local_index = np.where(minority_indices == idx)[0][0]
        
        if config['logging']['diagnostic']:
            print(f"Local index for minority sample at {idx} in the original data is : {local_index} ")

        neighbor_idxs = neighbor_indices[local_index]  # Self is already excluded from the neigh_indices object

        #print(f"self index for {idx} is {indices[0][1]}.")

        if config['logging']['diagnostic']:
            print(f"Neighbours of the original data {idx} and minority only data_idx {local_index} which will be used for synthetic obse {neighbor_idxs}.")

        neighbor_idxs = np.random.choice(neighbor_idxs, math.ceil(n_neighbors - 1), replace=False )   

        if config['logging']['diagnostic']:
            print(f"Neighbours of the original data {idx} and minority only data_idx {local_index} --> selected for synthetic obse {neighbor_idxs}.")
            print("NN of ",idx," selected for synthetic obs generation : ",neighbor_idxs)

        # Get neighbor samples (on full feature set)
        neighbors = x_minority[neighbor_idxs]  # FULL 2500-D feature vectors
        sel_minority_sample = x_minority[local_index]

        ## add the selected minority sample also to the array on which majority rank is done.
        combined = np.vstack([sel_minority_sample, neighbors])

        if config['logging']['diagnostic']:
            print(f"Select minority sample is {local_index} and the obs are {sel_minority_sample}.")
            print(f" full feature set of neighbour {neighbor_idxs} for observation {neighbors} :")

        ##################################################################################################
        ### Voting across the selected samples to generate the synthetic sample
        ##################################################################################################
        
        # Count how many samples support '1' per feature
        count_ones = np.sum(combined, axis=0)

        # Total number of samples considered
        total_samples = combined.shape[0]

        # Apply threshold rule
        synthetic_sample = (count_ones >= config['synthetic_sample_generation']['cut_off'] * total_samples).astype(int)
        
        if config['logging']['diagnostic']:
            print("cut off used for generation of synthetic sample :", config['synthetic_sample_generation']['cut_off'])
            print("synthetic sample generated :", synthetic_sample)

        synthetic_samples.append(synthetic_sample)
        
    
    # Append to training set
    x_synthetic = np.array(synthetic_samples)
    y_synthetic = np.ones(len(x_synthetic))

    x_resampled = np.vstack((x_train, x_synthetic))
    y_resampled = np.concatenate((y_train, y_synthetic))
    
    
    return x_resampled, y_resampled
    
    


