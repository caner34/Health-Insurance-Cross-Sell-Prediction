

import pandas as pd
import numpy as np
import math
import os


data = pd.read_csv(os.path.join('data', 'train.csv'))


##############################
##############################
####                      ####
####    PRE-PROCESSING    ####
####                      ####
##############################
##############################


data.columns
data.dtypes


######################################
######################################
####                              ####
####     Null Value Treatment     ####
####                              ####
######################################
######################################


# There is no null or missing value
info_null_missing_values = data.isnull().sum()



from pandas.api.types import is_string_dtype
def convert_data_into_numeric_types(data):
    data.Vehicle_Age = data.Vehicle_Age.apply(lambda x: 1 if x == '1-2 Year' else 0)
    data.Gender = data.Gender.apply(lambda x: 1 if x == 'Female' else 0)
    data.Vehicle_Damage = data.Vehicle_Damage.apply(lambda x: 1 if x == 'Yes' else 0)
    columns_to_be_converted = [x for x in data.columns if is_string_dtype(data[x])] + ['Region_Code', 'Policy_Sales_Channel', 'Vintage']
    data[columns_to_be_converted] = data[columns_to_be_converted].astype(object)
    
    return data


# apply logarithmic function to required features 
# to handle outliers with astronomical values
def shrink_data_logarithmically(data, feature_to_be_shrunk):
    data[feature_to_be_shrunk] = data[feature_to_be_shrunk].apply(lambda x: math.log10(x))
    return data


from sklearn.preprocessing import MinMaxScaler

# normalizes numeric features for faster computation and possiblly better performance
def normalize_numeric_data(data):
    min_max_scaler = MinMaxScaler()
    features_to_be_normalized = [x for x in data.columns if not is_string_dtype(data[x])]
    data[features_to_be_normalized] = min_max_scaler.fit_transform(data[features_to_be_normalized])
    return data


# Create dummy variables for categorical features
def get_data_with_dummies(data):
    categorical_features = [x for x in data.columns if is_string_dtype(data[x])]
    df_dummies = pd.get_dummies(data[categorical_features])
    data = pd.merge(data, df_dummies, how="inner", left_index=True, right_index=True).drop(columns=categorical_features)
    
    # drop the last dummy column for each category group since its redundant
    redundant_features = [[x for x in data.columns if x.startswith(c)][-1] for c in categorical_features]
    data = data.drop(columns=redundant_features)
    return data


def execute_preprocessing(data):
    
    data = data.drop(columns=['id'])
    data = convert_data_into_numeric_types(data)
    data = shrink_data_logarithmically(data, 'Annual_Premium')
    data = normalize_numeric_data(data)
    data = get_data_with_dummies(data)
    
    return data


# data.Annual_Premium.describe()



