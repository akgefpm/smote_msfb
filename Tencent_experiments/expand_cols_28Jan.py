
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix
import pickle
import sys
import gc

import snappy
import fastparquet

import zipfile
import pandas as pd
from io import StringIO

from sklearn.preprocessing import LabelEncoder
import time

def expand_cols_28Jan(data, column_name):
    
    print("Shape of the input dataset :", data.shape)
          
    data_input = data[ ['aid','uid',column_name] ].copy()
    
    print("Shape of the input dataset :", data_input.shape)
        
    # Record start time
    start_time = time.time() 
    
    # 1. Convert the 'interest5' column to a list of integers
    data_input[column_name] = data_input[column_name].apply(lambda x: [int(i) for i in x.split()] if x is not None else [])
    
    end_time = time.time()
    # Calculate the time taken
    time_taken = end_time - start_time
    print(f"Time taken for conversion of string to numeric list for each row takes : {time_taken} seconds")
    
    ## 2. Explode the column_name into separate rows
    data_input_exploded = data_input.set_index(['uid', 'aid'])[column_name].explode().reset_index()

    start_time = time.time() 
    
    end_time = time.time()
    # Calculate the time taken
    time_taken = end_time - start_time
    print(f"Time taken for the step. 2 (exploding the list into separate rows) takes : {time_taken} seconds")

    start_time = time.time()

    ## 3. Encode the unique values using label encoder
    le = LabelEncoder()
    column_name_encoded = (column_name +'_encoded')
    data_input_exploded[column_name_encoded] = le.fit_transform(data_input_exploded[column_name])
    
    print("Total no. of unique columns added to data table: ", len(le.classes_))

    end_time = time.time()
    # Calculate the time taken
    time_taken = end_time - start_time
    print(f"Time taken for the step. 3 (label encoding) : {time_taken} seconds")

    start_time = time.time()

    ## 3. Pivot the data to create the unique columns
    data_input_pivot = data_input_exploded[ ['uid', 'aid', column_name_encoded] ].pivot_table(index=['uid', 'aid'], columns= column_name_encoded, 
                                       aggfunc=lambda x: 1, fill_value=0)

    end_time = time.time()
    # Calculate the time taken
    time_taken = end_time - start_time
    print(f"Time taken for the step. 4 (Pivoting the data) takes : {time_taken} seconds")

    print("Shape of the output dataset (shape of pivoted data) :", data_input_pivot.shape)
    
    start_time = time.time()
    
    ## Step 5. Add the new columns to the original dataset
    column_name_header = (column_name +'_')
    data_input_pivot.columns = [ column_name_header + str(col) for col in data_input_pivot.columns]
    
    data_input_pivot.reset_index(inplace=True)
    
    print("Shape of the output dataset :", data_input_pivot.shape)
    
    #data_input_pivot['uid'] = data_input_pivot['uid'].astype(int)
    #data_input_pivot['aid'] = data_input_pivot['aid'].astype(int)
    
    data = data.merge(data_input_pivot, how="inner", on=['uid','aid'])
    
    print("Shape of the output dataset (after merging pivoted columns) :", data.shape)
    
    end_time = time.time()
    # Calculate the time taken
    time_taken = end_time - start_time
    print(f"Time taken for the step. 5 : {time_taken} seconds")
    
    data.drop(column_name, axis=1, inplace=True) 
    
    print("Shape of the output dataset (after dropping original column) :", data.shape)
    
    return data