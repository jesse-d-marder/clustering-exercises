import pandas as pd
import numpy as np
from env import get_db_url
import os
from sklearn.model_selection import train_test_split
import sklearn.preprocessing

def acquire_mall():
    """ Acquire mall customer data from mySQL server or cached csv, returns as Pandas DataFrame """
    
    if os.path.exists('mall_customers.csv'):
        print("Using cached data")
        mall = pd.read_csv('mall_customers.csv')
    # Acquire data from database if CSV does not exist
    else:
        print("Acquiring data from server")
        query = """ SELECT * FROM customers; """
        mall = pd.read_sql(query, get_db_url('mall_customers'))
        
        mall.to_csv('mall_customers.csv', index=False)
        
    
    return mall

def wrangle_mall():
    """ Acquires, encodes gender, splits, and scales data from mall customers. Returns 3 dataframes """
    
    mall = acquire_mall()
    
    # Encode gender
    mall = pd.get_dummies(mall, columns=['gender'], drop_first=True)
    
    # Split data
    
    train, validate, test = split_data(mall)
    
    # Scale data
    
    train_scaled, validate_scaled, test_scaled = scale_data(train, validate, test, ['age','annual_income','spending_score'])
    
    return train_scaled, validate_scaled, test_scaled

def split_data(df, train_size_vs_train_test = 0.8, train_size_vs_train_val = 0.7, random_state = 123):
    """Splits the inputted dataframe into 3 datasets for train, validate and test (in that order).
    Can specific as arguments the percentage of the train/val set vs test (default 0.8) and the percentage of the train size vs train/val (default 0.7). Default values results in following:
    Train: 0.56
    Validate: 0.24
    Test: 0.2"""
    train_val, test = train_test_split(df, train_size=train_size_vs_train_test, random_state=123)
    train, validate = train_test_split(train_val, train_size=train_size_vs_train_val, random_state=123)
    
    train_size = train_size_vs_train_test*train_size_vs_train_val
    test_size = 1 - train_size_vs_train_test
    validate_size = 1-test_size-train_size
    
    print(f"Data split as follows: Train {train_size:.2%}, Validate {validate_size:.2%}, Test {test_size:.2%}")
    
    return train, validate, test

def scale_data(train, validate, test, features_to_scale):
    """Scales data using MinMax Scaler. 
    Accepts train, validate, and test datasets as inputs as well as a list of the features to scale. 
    Returns dataframe with scaled values added on as columns"""
    
    # Fit the scaler to train data only
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(train[features_to_scale])
    
    # Generate a list of the new column names with _scaled added on
    scaled_columns = [col+"_scaled" for col in features_to_scale]
    
    # Transform the separate datasets using the scaler learned from train
    scaled_train = scaler.transform(train[features_to_scale])
    scaled_validate = scaler.transform(validate[features_to_scale])
    scaled_test = scaler.transform(test[features_to_scale])
    
    # Concatenate the scaled data to the original unscaled data
    train_scaled = pd.concat([train, pd.DataFrame(scaled_train,index=train.index, columns = scaled_columns)],axis=1)
    validate_scaled = pd.concat([validate, pd.DataFrame(scaled_validate,index=validate.index, columns = scaled_columns)],axis=1)
    test_scaled = pd.concat([test, pd.DataFrame(scaled_test,index=test.index, columns = scaled_columns)],axis=1)

    return train_scaled, validate_scaled, test_scaled

def remove_outliers(df, k, col_list):
    ''' Removes outliers based on multiple of IQR. Accepts as arguments the dataframe, the k value for number of IQR to use as threshold, and the list of columns. Outputs a dataframe without the outliers.
    '''
    # Create a column that will label our rows as containing an outlier value or not
    num_obs = df.shape[0]
    df['outlier'] = False
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # update the outlier label any time that the value is outside of boundaries
        df['outlier'] = np.where(((df[col] < lower_bound) | (df[col] > upper_bound)) & (df.outlier == False), True, df.outlier)
    
    df = df[df.outlier == False]
    df.drop(columns=['outlier'], inplace=True)
    print(f"Number of observations removed: {num_obs - df.shape[0]}")
        
    return df