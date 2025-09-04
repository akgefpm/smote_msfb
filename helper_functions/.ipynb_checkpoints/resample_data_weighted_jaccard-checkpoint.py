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


def resample_data_weighted_jaccard(x_train, y_train, minority_resample_list, no_of_synthetic_samples_to_be_generated, n_neighbors=5, perc_MI = 0.95):
    """
    
    Generates synthetic samples for the given minority_resample_list based on 5-NN using weighted Jaccard distance.
    To improve robustness in high-dimensional binary data, feature selection using mutual information is performed
    for neighbor search only.
    """   
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    minority_indices = np.where(y_train == 1)[0]
    x_minority = x_train[minority_indices] 
    
    ### Part 1 -> The below part selects top features from the data which contribute to 95% of the Mutual Information.

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

    print(f"Selected {num_features_to_select} features that account for {perc_MI*100:.1f}% of total mutual information.")
    
    # Use only top features for neighbor computation
    x_minority_selected = x_minority[:, top_features]
    
    print("Shape of x_minority_selected :", x_minority_selected.shape )
    
    # Use MI scores of selected features as weights
    weights = mi_scores[top_features]
    
    # Normalize the weights to sum to 1
    weights = weights / np.sum(weights)

    # Compute weighted Jaccard distance for all minority to minority data points
    jaccard_dist = weighted_jaccard_distance(x_minority_selected, weights)
    
    print("weights.shape :", weights.shape)
    
    # Fit KNN on minority samples (on selected features)
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='precomputed')  # +1 to exclude self
    knn.fit(jaccard_dist)    
    
    print("Shape of jaccard_dist matrix :", jaccard_dist.shape ) ## This is equal to minority * minority matrix
    
    ### The below part checks for the hubness of the data 
    # ---- HUBNESS CHECK ----
    print("Checking for hubness phenomenon among minority samples...")

    # For each minority sample, find its neighbors
    neighbor_indices = knn.kneighbors(jaccard_dist, return_distance=False)

    # Exclude self-neighbors (first neighbor is the point itself because distance = 0)
    neighbor_indices = neighbor_indices[:, 1:]
    
    # Count how many times each sample appears in others' neighbor lists
    hubness_counts = np.zeros(x_minority.shape[0], dtype=int)

    for neighbors in neighbor_indices:
        for idx in neighbors:
            hubness_counts[idx] += 1

    # Now we have the hubness count (number of times each sample is a neighbor)
    # Let's print summary stats
    print("Hubness Stats:")
    print(f"  Max times a sample appeared as neighbor: {hubness_counts.max()}")
    print(f"  Min times a sample appeared as neighbor: {hubness_counts.min()}")
    print(f"  Mean times a sample appeared as neighbor: {hubness_counts.mean():.2f}")
    print(f"  Std deviation: {hubness_counts.std():.2f}")

    # Create and print PDF (normalized histogram)
    unique_counts, frequencies = np.unique(hubness_counts, return_counts=True)
    pdf = frequencies / frequencies.sum()

    print("Hubness PDF (Times as Neighbor -> Probability):")
    for count, prob in zip(unique_counts, pdf):
        print(f"  {count}: {prob:.4f}")

    # Plotting the PDF
    plt.figure(figsize=(8, 5))
    plt.bar(unique_counts, pdf, width=0.6, color='skyblue', edgecolor='black')
    plt.xlabel("Number of times a sample appears as a neighbor")
    plt.ylabel("Probability")
    plt.title("Hubness PDF (Minority Samples)")
    plt.xticks(unique_counts)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Resampling strategy
    print("No. of synthetic samples to be generated:", no_of_synthetic_samples_to_be_generated)

    if no_of_synthetic_samples_to_be_generated > len(minority_resample_list):
        print("----------- RESAMPLE DATA - NO. OF SYNTHETIC SAMPLES TO BE GENERATED IS HIGHER THAN MISSCLASSIFIED OBSERVATIONS -------------")

    print(f"We are drawing {no_of_synthetic_samples_to_be_generated} samples from an array of size {len(minority_resample_list)}.")
    
    ## Sample evenly across the minority samples identified - While choosing from the miss-classified minority samples, 
    ## a larger set of minority samples for creating 
    ## synthetic observations. We will first create synthetic samples from all miss-classified samples evenly as much as possible and 
    ## then only for the remainder we will use randomness.

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
        
    synthetic_samples = []

    for idx in minority_resample_list_upd:
        # Get index of this sample in the minority array

        print("Minority Sample for which synthetic obs are generated :", idx)

        local_index = np.where(minority_indices == idx)[0][0]
        print(f"Local index for minority sample at {idx} in the original data is : {local_index} ")

        # Get neighbors (excluding self)
        # distances, indices = knn.kneighbors([jaccard_dist[local_index]])

        neighbor_idxs = neighbor_indices[local_index]  # Self is already excluded from the neigh_indices object

        #print(f"self index for {idx} is {indices[0][1]}.")

        print(f"Neighbours of the original data {idx} and minority only data_idx {local_index} which will be used for synthetic obse {neighbor_idxs}.")

        neighbor_idxs = np.random.choice(neighbor_idxs, math.ceil(n_neighbors / 2), replace=False )   

        print(f"Neighbours of the original data {idx} and minority only data_idx {local_index} --> selected for synthetic obse {neighbor_idxs}.")

        #print("NN of ",idx," selected for synthetic obs generation : ",neighbor_idxs)

        # Get neighbor samples (on full feature set)
        neighbors = x_minority[neighbor_idxs]  # FULL 2500-D feature vectors
        sel_minority_sample = x_minority[local_index]

        ## add the selected minority sample also to the array on which majority rank is done.
        combined = np.vstack([sel_minority_sample, neighbors])

        print(f"Select minority sample is {local_index} and the obs are {sel_minority_sample}.")

        print(f" full feature set of neighbour {local_index} for observation {neighbors} :")

        # Majority vote across columns (binary features)
        # synthetic_sample = (np.sum(neighbors, axis=0) >= (n_neighbors // 2 + 1)).astype(int) # <--- majority voting across the neighbours
        synthetic_sample = (np.sum(combined, axis=0) > 0).astype(int) ## <----- max voting across the neighbours

        print("synthetic sample generated :", synthetic_sample)

        synthetic_samples.append(synthetic_sample)
        
    
    # Append to training set
    x_synthetic = np.array(synthetic_samples)
    y_synthetic = np.ones(len(x_synthetic))

    x_resampled = np.vstack((x_train, x_synthetic))
    y_resampled = np.concatenate((y_train, y_synthetic))
    
    
    return x_resampled, y_resampled
    
    


