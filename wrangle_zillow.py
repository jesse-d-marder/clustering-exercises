import pandas as pd
import numpy as np
from env import get_db_url
import os
from sklearn.model_selection import train_test_split
import sklearn.preprocessing

def wrangle_zillow():
    """ Acquires the Zillow housing data from the SQL database or a cached CSV file. Renames columns and outputs data as a Pandas DataFrame"""
    # Acquire data from CSV if exists
    if os.path.exists('zillow_2017.csv'):
        print("Using cached data")
        df = pd.read_csv('zillow_2017.csv')
    # Acquire data from database if CSV does not exist
    else:
        print("Acquiring data from server")
        query = """
                SELECT * FROM properties_2017
                LEFT JOIN (SELECT logerror, transactiondate, parcelid AS parcelid_pred FROM predictions_2017) as preds_2017
                ON preds_2017.parcelid_pred = properties_2017.parcelid
                LEFT JOIN propertylandusetype as prop_type
                USING (propertylandusetypeid)
                LEFT JOIN airconditioningtype as ac_type
                USING (airconditioningtypeid)
                LEFT JOIN architecturalstyletype as arch_type
                USING (architecturalstyletypeid)
                LEFT JOIN buildingclasstype as b_class_type
                USING (buildingclasstypeid)
                LEFT JOIN heatingorsystemtype as heat_type
                USING (heatingorsystemtypeid)
                LEFT JOIN storytype
                USING (storytypeid)
                LEFT JOIN typeconstructiontype
                USING (typeconstructiontypeid)
                INNER JOIN
                (SELECT parcelid AS parcel_id_max_date, MAX(transactiondate) as max_date
                FROM predictions_2017
                GROUP BY parcel_id_max_date) AS latest_transaction
                ON latest_transaction.parcel_id_max_date = preds_2017.parcelid_pred AND latest_transaction.max_date = preds_2017.transactiondate
                WHERE transactiondate IS NOT NULL
                AND longitude IS NOT NULL
                AND latitude IS NOT NULL
                AND transactiondate BETWEEN '2017-01-01' and '2017-12-31';
            """
        df = pd.read_sql(query, get_db_url('zillow'))
        
        # Drop unnecessary foreign keys
        df = df.drop(columns = ['parcel_id_max_date','max_date','parcelid_pred', 'typeconstructiontypeid','storytypeid','heatingorsystemtypeid','buildingclasstypeid','architecturalstyletypeid','architecturalstyletypeid','airconditioningtypeid','propertylandusetypeid'])
        df.to_csv('zillow_2017.csv', index=False)
    
    # Prepare the data for exploration and modeling
    # Rename columns as needed
    df=df.rename(columns = {'bedroomcnt':'bedroom', 
                            'bathroomcnt':'bathroom', 
                            'calculatedfinishedsquarefeet':'square_feet',
                            'taxvaluedollarcnt':'tax_value',
                            'garagecarcnt':'garage',
                           'buildingqualitytypeid':'condition',
                           'regionidzip':'zip',
                           'poolcnt':'pools',
                           'lotsizesquarefeet':'lot_size'})
    
    
    return df

def handle_missing_zillow_values(df):
    """ Specific to Zillow dataset. Filters to single unit properties and deals with null values."""
    
    # Just want single unit properties
    
    # Filter out anything other than unit count = 1 and nans
    df = df[(df.unitcnt==1)|(df.unitcnt.isna())]
    
    # Keep properties that should be single units
    properties_to_keep = ['Single Family Residential','Condominum','Mobile Home','Manufactured, Modular, Prefabricated Homes','Residential General','Townhouse',np.nan]
    df = df[df.apply(lambda row: row.propertylandusedesc in properties_to_keep, axis=1)]
    
    # First pass on removing missing values
    df_nulls_removed = handle_missing_values(df, prop_required_column=0.3, prop_required_row=0.00002)
    
    # Fill na values and drop specific columns based on exploring nans
    df_nulls_removed['garage'] = df_nulls_removed['garage'].fillna(0)
    df_nulls_removed['garagetotalsqft'] = df_nulls_removed['garagetotalsqft'].fillna(0) # 'No garage'
    df_nulls_removed['poolsizesum'] = df_nulls_removed['poolsizesum'].fillna(0)# 'No pool'
    df_nulls_removed['basementsqft'] = df_nulls_removed['basementsqft'].fillna(0) # 'No basement information'
    df_nulls_removed['threequarterbathnbr'] = df_nulls_removed['threequarterbathnbr'].fillna(0)
    df_nulls_removed['taxdelinquencyyear'] = df_nulls_removed['taxdelinquencyyear'].fillna(0) # "Assumed Not Delinquent"
    df_nulls_removed['condition'] = df_nulls_removed['condition'].fillna(-1) # "Not available"
    df_nulls_removed['yardbuildingsqft17'] = df_nulls_removed['yardbuildingsqft17'].fillna(0) # "No Patio Information"
    df_nulls_removed['yardbuildingsqft26'] = df_nulls_removed['yardbuildingsqft26'].fillna(0) # "No Yard Building"
    df_nulls_removed = df_nulls_removed.drop(columns = ['regionidneighborhood','calculatedbathnbr','finishedsquarefeet13','finishedsquarefeet50','finishedsquarefeet6','finishedsquarefeet12','finishedfloor1squarefeet'])
    # Make a column for the county based on FIPS
    df_nulls_removed["county"] = np.select([df_nulls_removed.fips == 6037, df_nulls_removed.fips==6059, df_nulls_removed.fips == 6111],["Los Angeles County", "Orange County", "Ventura County"])
    
    # Fill in binary values with 0s
    for col in df_nulls_removed.columns:
        if df_nulls_removed[col].nunique() == 1:
            df_nulls_removed[col] = df_nulls_removed[col].fillna('None')
    
    # Fill in count, number, and desc values with 0s and not specified
    for col in df_nulls_removed.columns:
        if 'desc' in col:
            df_nulls_removed[col] = df_nulls_removed[col].fillna('Not Specified')
        elif 'cnt' in col:
            df_nulls_removed[col] = df_nulls_removed[col].fillna(0)
        elif 'number' in col:
            df_nulls_removed[col] = df_nulls_removed[col].fillna(0)
            
    # For now, just remove remaining null values
    df_nulls_removed = df_nulls_removed.dropna()
            
    return df_nulls_removed

def handle_missing_values(df, prop_required_column, prop_required_row):
    """ Drops columns and rows from df that have fewer values than required by the 
    arguments prop_required_column and prop_required_row. First drops columns then drops rows. 
    Returns a df without the columns and rows that were dropped. """
    
    # Drop columns with pct of missing rows above threshold
    print("For threshold based dropping: ")
    print(df.shape, " original shape")
    df = df.dropna(thresh = int((prop_required_row)*len(df)), axis=1, inplace=False)
    print(df.shape, " shape after dropping columns with prop required rows below theshold")
    
    # Drop rows with pct of missing columns above threshold
    df = df.dropna(thresh = int(prop_required_column*len(df.columns)), inplace=False)
    print(df.shape, " shape after dropping rows with prop required columns below threshold")
    
    return df


def nulls_by_row(df):
    """ Returns the number of and percent of nulls per row, as well as the number of rows with the given missing num of columns """
    # nulls by row
    info =  pd.concat([
        df.isna().sum(axis=1).rename('num_cols_missing'),
        df.isna().mean(axis=1).rename('pct_cols_missing'),
    ], axis=1)
    
    return pd.DataFrame(info.value_counts(),columns = ['num_rows']).reset_index().sort_values(by='num_rows', ascending=False)

def nulls_by_column(df):
    """ Returns the number of and percent of nulls per column """
    return pd.concat([
        df.isna().sum(axis=0).rename('n_rows_missing'),
        df.isna().mean(axis=0).rename('pct_rows_missing'),
    ], axis=1).sort_values(by='pct_rows_missing', ascending=False)

def get_upper_outliers(s, k):
    '''
    Given a series and a cutoff value, k, returns the upper outliers for the
    series.
    
    The values returned will be either 0 (if the point is not an outlier), or a
    number that indicates how far away from the upper bound the observation is.
    '''
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    upper_bound = q3 + k * iqr
    return s.apply(lambda x: max([x - upper_bound, 0]))

def get_lower_outliers(s, k):
    '''
    Given a series and a cutoff value, k, returns the lower outliers for the
    series.
    
    The values returned will be either 0 (if the point is not an outlier), or a
    number that indicates how far away from the lower bound the observation is.
    '''
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    return s.apply(lambda x: max([x - lower_bound, 0]))

def add_upper_outlier_columns(df, k, describe=False):
    '''
    Add a column with the suffix _upper_outliers for all the numeric columns
    in the given dataframe and the given cutoff k value. Optionally displays a description of the outliers.
    '''
    
    for col in df.select_dtypes('number'):
        df[col + '_upper_outliers'] = get_upper_outliers(df[col], k)
    
    outlier_cols = [col for col in df if col.endswith('_upper_outliers')]

    if describe:
        for col in outlier_cols:
            print('---\n' + col)
            data = df[col][df[col] > 0]
            print(data.describe())

    return df

def add_lower_outlier_columns(df, k, describe = False):
    '''
    Add a column with the suffix _lower_outliers for all the numeric columns
    in the given dataframe and the given cutoff k value. Optionally displays a description of the outliers.
    '''
    
    for col in df.select_dtypes('number'):
        df[col + '_lower_outliers'] = get_lower_outliers(df[col], k)
    
    outlier_cols = [col for col in df if col.endswith('_lower_outliers')]

    if describe:
        for col in outlier_cols:
            print('---\n' + col)
            data = df[col][df[col] > 0]
            print(data.describe())
            
    return df

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