
import numpy as np
from get_minority_obs_to_resample import get_minority_obs_to_resample
from resample_data import resample_data

def smote_msfb(x_train, y_train, 
               config):
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)         
    
    resample_idx, sample_probs_global = get_minority_obs_to_resample(x_train, y_train, config)
    
    x_train_upd, y_train_upd = resample_data(x_train, y_train, 
                                             resample_idx, 
                                             config) 
    
    return x_train_upd, y_train_upd