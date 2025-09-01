
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import Binarizer
import random

def generate_imbalanced_data(n_rows, imbalance_ratio, n_columns, flip_y, class_sep, informative_feature_perc, random_seed=42):
    """
    Generates an imbalanced binary classification dataset with correlated and noisy features.
    
    Parameters:
    - n_rows (int): Number of rows (samples).
    - imbalance_ratio (float): Proportion of the minority class (e.g., 0.1 means 10% minority class).
    - n_columns (int): Number of features (covariates).
    - random_seed (int): Random seed for reproducibility.
    
    Returns:
    - pd.DataFrame: DataFrame with binary features and the target variable.
    """
    random_seed = random.randint(10, 1000)
    np.random.seed(random_seed)
        
    # Generate correlated features with some noise
    X, y = make_classification(
        n_samples=n_rows, 
        n_features=n_columns, 
        n_informative=int(n_columns * informative_feature_perc), # 70% informative features
        n_redundant=int(n_columns * (1 - informative_feature_perc)),  # 20% redundant features (correlated with informative)
        n_classes=2, 
        weights=[1 - imbalance_ratio, imbalance_ratio],  # Class imbalance
        flip_y=flip_y,  # No noise in target variable
        class_sep = class_sep,
        random_state=random_seed
    )
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(n_columns)])
    
    # Add the target variable
    df['target'] = y
    
    # Binarize the noisy features (convert them to binary)
    binarizer = Binarizer()
    for col in df.columns[:-1]:  # Apply to all features except the target column
        df[col] = binarizer.fit_transform(df[col].values.reshape(-1, 1))
    
    if flip_y == 0:
        
        # Assuming your dataset is in a pandas DataFrame `df` with columns 'f1' to 'f200' and a 'target' column

        # Step 1: Split the dataset into majority and minority classes based on the 'target' column
        majority_class = df[df['target'] == 0]
        minority_class = df[df['target'] == 1]

        # Step 2: Flip 50% of the minority class observations to target 0
        num_to_flip_minority = len(minority_class) // 5  # 20% of minority class
        print("We are introducing noise in the seed list by flipping 20% of the seed observations.")
        minority_class_flipped = minority_class.head(num_to_flip_minority)  # Take the first 50% (or randomly, if needed)

        # Change their target labels to 0
        minority_class_flipped['target'] = 0

        # Step 3: Flip an equal number of observations from the majority class to target 1
        num_to_flip_majority = num_to_flip_minority  # Equal number as the minority class flipped
        majority_class_flipped = majority_class.head(num_to_flip_majority)  # Take the first 50% (or randomly, if needed)

        # Change their target labels to 1
        majority_class_flipped['target'] = 1

        # Step 4: Recombine the modified majority and minority classes with the rest of the dataset
        # Add the flipped minority and majority rows back to the original data
        modified_df = pd.concat([df, minority_class_flipped, majority_class_flipped])

        # Step 5: Drop the original rows that were flipped from the majority and minority classes
        df_remaining = pd.concat([majority_class.tail(len(majority_class) - num_to_flip_majority), 
                                  minority_class.tail(len(minority_class) - num_to_flip_minority)])

        # Combine everything back
        final_df = pd.concat([df_remaining, minority_class_flipped, majority_class_flipped])

        # Step 6: Shuffle the entire dataset again to ensure randomness
        final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Now `final_df` is your dataset with noise added but keeping the class imbalance intact     
    else:
        final_df = df
        
    # Return the dataset
    return final_df