
import numpy as np
#from resample_data import resample_data
# from get_minority_obs_to_resample import get_minority_obs_to_resample
from get_minority_obs_to_resample_bagging_filtered import get_minority_obs_to_resample_bagging_filtered
from resample_data_weighted_jaccard import resample_data_weighted_jaccard

def minority_focussed_smotten(x_train, y_train, no_of_synthetic_samples_to_be_generated, neighbours_to_consider_for_neighbourhood = 5):
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    # resample_idx = get_minority_obs_to_resample(x_train, y_train)    
    
    resample_idx, sample_probs_global = get_minority_obs_to_resample_bagging_filtered(x_train, y_train)
    
    if no_of_synthetic_samples_to_be_generated == 0:
        no_of_synthetic_samples_to_be_generated = (np.bincount(y_train)[0] - np.bincount(y_train)[1])
    
    x_train_upd, y_train_upd = resample_data_weighted_jaccard(x_train, y_train, 
                                             resample_idx, 
                                             no_of_synthetic_samples_to_be_generated,
                                             n_neighbors= neighbours_to_consider_for_neighbourhood)
    
    return x_train_upd, y_train_upd