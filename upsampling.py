import pandas as pd
import numpy as np
from sklearn.utils import resample

def up_sample_minority_class(X, y, target_feature, scale=1.0):
    
    data = X.copy()
    # print("\n y: "+y)
    data[target_feature] = y
    
    
    # Entries of the both minority and majority classes
    value_majority = data[target_feature].value_counts().sort_values(ascending=False).index[0]
    data_majority = data.loc[data[target_feature] == value_majority]
    data_minority = data.loc[data[target_feature] != value_majority]
    
    # print("data_majority: {0} @ data_minority: {1}".format(len(data_majority), len(data_minority)))
    
    if scale > 1.0:
        scale = 1.0
    elif scale < 1.0:
        data_majority = resample(data_majority, 
                                     replace=True,
                                     n_samples=int(len(data_majority)*scale),
                                     random_state=13456)
    
    #populates the minority portion of the samples up to the size of majority portion
    data_minority_up_sampled = resample(data_minority, 
                                     replace=True,
                                     n_samples=len(data_majority),
                                     random_state=13456)
    
    # Combine majority class with upsampled minority class
    data_up_sampled = pd.concat([data_majority, data_minority_up_sampled])
    
    # Display new class counts
    # print(data_up_sampled[target_feature].value_counts())
    
    data_up_sampled = data_up_sampled.reset_index(drop=True)
    
    X = data_up_sampled.drop(columns=[target_feature])
    y = data_up_sampled[target_feature]
    
    return X, y
