import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline,make_pipeline


"""
    parameter:
        df: pandas dataframe
"""

def delete_duplicates(df, keep = 'first'):
    if isinstance(df, pd.DataFrame):
        df.drop_duplicates(keep=keep, inplace=True)
        return
    raise ValueError(f"Invalid datatype {type(df)} this function accept pandas dataframe")



"""
    parameter:
        df: pandas dataframe
"""

def make_category_columns(df, min_cat_percent = 5.0):
    if isinstance(df, pd.Series):
        df = df.to_frame()

    columns = df.columns

    for cols in columns:
        if pd.api.types.is_object_dtype(df[cols]):
            count = df[cols].nunique()
            percent_count = (count/df[cols].shape[0])*100 

            if percent_count <= min_cat_percent:
                df[cols] = df[cols].astype('category')
            else:
                df.drop(cols, axis=1, inplace=True)
    return 

"""
    function ends
"""

# CCA--> Complete case Analysis

"""
1. If there is no null values in dataframe then simply return 
2. It there is columns that have null values less than 4% then drop
   the rows from that columns
   
   2.1 If total dropped data is less than 10% then its fine else don't
   drop the rows return original data.
"""


def drop_row_column(df, datalosspercent = 10, min_var = 0.04):
    if isinstance(df, pd.DataFrame) == False:
        raise ValueError(f"Invalid datatype {type(df)} this function accept pandas dataframe")

    null_count = df.isnull().sum().values.sum()
    
    for col in df.columns:
        if df[col].nunique() == df.shape[0]:
            df.drop(col, axis=1, inplace=True)

    if null_count == 0: # if data is completely clean
        return
    else:
        limited_null_columns = [var for var in df.columns if df[var].isnull().mean() <= min_var]
        excessive_null_columns = [var for var in df.columns if df[var].isnull().mean() > min_var]

        if len(limited_null_columns) == 0:
            return   
        else:
            new_df = df[limited_null_columns].dropna()
            new_df[excessive_null_columns] = df[excessive_null_columns]

            change = ((df.shape[0]-new_df.shape[0])/df.shape[0])*100

            if change > datalosspercent:
                return 
            else:
                df = new_df
                return

# Univariate Imputation 

"""
 parameters :
    df: pandas dataframe accepted as data
    limit: upper limit of data that can be null, so that can be processed (range should 0 to 1)
    var_diff: after imputation the upper limit to which the variance of the column may change ( range sould 0 to 1)
    mod_diff: after imputation the upper limit to which the mode of the column may change (range should be 0 to 1)
 purpose: 
    univariate imputation on numerical column and categorical column
"""

def impute_row_column(df, min_var = 0.04, var_diff = 0.05, mod_diff = 0.05, numerical_column = [], categorical_column = [], temporal_column = [], temporal_type = 'interpolate'):

    """
        error handling
    """
    if isinstance(df, pd.DataFrame) == False:
        raise ValueError(f"Invalid datatype {type(df)} this function accept pandas dataframe")

    if(min_var > 1 or min_var < 0):
        raise ValueError("Invalid value. Value of limit should be from 0 to 1 inclusive.")
    if(var_diff > 1 or var_diff < 0):
        raise ValueError("Invalid value. Value of var_diff should be from 0 to 1 inclusive.")
    if(mod_diff > 1 or mod_diff < 0):
        raise ValueError("Invalid value. Value of mod_diff should be from 0 to 1 inclusive.")
    
    dtype_list = ['object', 'category', 'string', 'interval', 'bool', 'int64', 'int32', 'int16', 'int8', 'uint64', 'uint32', 'uint16', 'uint8', 'float64', 'float32', 'datetime64[ns]', 'timedelta64[ns]', 'datetime64[ns, tz]', 'period']
    
    for i in df.columns:
        if df[i].dtype not in dtype_list:
            raise ValueError(f"Invalid dtype, this module can work with only {dtype_list} dtypes")


    null_count = df.isnull().sum().values.sum()
    if null_count == 0: # if data is completely clean
        return
    else:
        # columns that have less than given null values limit
        limited_null_columns = [var for var in df.columns if df[var].isnull().mean() <= min_var]
        
        if len(limited_null_columns) == 0:
            return
        else:
            # checking for categorical column
            
            if len(categorical_column) == 0:
                category_type = ['object', 'category', 'string', 'interval', 'bool']
                categorical_column = [col for col in df.columns if col in limited_null_columns and (df[col].dtype in category_type)]
            if len(numerical_column) == 0:
                numeric_type = ['int64', 'int32', 'int16', 'int8', 'uint64', 'uint32', 'uint16', 'uint8', 'float64', 'float32']
                numerical_column = [col for col in df.columns if col in limited_null_columns and (df[col].dtype in numeric_type)]
            if len(temporal_column) == 0:
                time_type = ['datetime64[ns]', 'timedelta64[ns]', 'datetime64[ns, tz]', 'period']
                temporal_column = [col for col in df.columns if col in limited_null_columns and (df[col].dtype in time_type)]
            
            imputer1 = SimpleImputer(strategy='median')
            imputer2 = SimpleImputer(strategy='mean')
            imputer4 = SimpleImputer(strategy='most_frequent')

            trf1 = ColumnTransformer([
                ('imputer1',imputer1,numerical_column)
            ],remainder='passthrough')
            
            trf2 = ColumnTransformer([
                ('imputer2', imputer2, numerical_column)
            ], remainder='passthrough')

            trf4 = ColumnTransformer([
                ('imputer4', imputer4, categorical_column)
            ], remainder='passthrough')

            new_data = df # temporary data
            variation_before_imputation = df[numerical_column].var()
            
            median = trf1.fit_transform(new_data[numerical_column])
            mean = trf2.fit_transform(new_data[numerical_column])
            mode = trf4.fit_transform(new_data[categorical_column])

            # handling of numerical column

            for i in range(len(numerical_column)):
                percentage_change_median = ((variation_before_imputation.iloc[i]-median[:, i].var())/variation_before_imputation.iloc[i])
                percentage_change_mean = ((variation_before_imputation.iloc[i]-mean[:, i].var())/variation_before_imputation.iloc[i])
                
                if (percentage_change_median > percentage_change_mean) and (percentage_change_mean < var_diff):
                    new_data[numerical_column[i]] = mean[:, i]
                elif percentage_change_median < var_diff:
                    new_data[numerical_column[i]] = median[:, i]
                else:
                    new_data[numerical_column[i]] = df[numerical_column[i]]

            # handling of categorical column
            
            for i in range(len(categorical_column)):
                values = df[categorical_column[i]].value_counts().sort_values(ascending=False)
                if values.iloc[1]/values.iloc[0] > mod_diff:
                    continue
                else:
                    new_data[categorical_column[i]] = mode[:, i]
                
            # handling of temporal column

            for i in range(len(temporal_column)):
                if temporal_type == 'bfill':
                    new_data[temporal_column[i]] = df[temporal_column[i]].bfill()
                elif temporal_type == 'ffill':
                    new_data[temporal_column[i]] = df[temporal_column[i]].ffill()
                elif temporal_type == 'interpolate':
                    new_data[temporal_column[i]] = df[temporal_column[i]].interpolate()
                else:
                    raise ValueError("incorrect method used")
            
            df = new_data
            return

# multivariate knn Imputer

"""
 parameters :
    df: pandas dataframe accepted as data
    n_neighbors: number of neighbors for imputing data
 purpose: 
    multivariate imputation on numerical column and categorical column
"""



def multivariate_impute(df, n_neighbors = 5):
    if isinstance(df, pd.DataFrame) == False:
        raise ValueError(f"Invalid datatype {type(df)} this function accept pandas dataframe")
    
    numeric_type = ['int64', 'int32', 'int16', 'int8', 'uint64', 'uint32', 'uint16', 'uint8', 'float64', 'float32']
    null_numeric_columns = [col for col in df.columns if (df[col].isnull().sum() > 0) and (df[col].dtype in numeric_type)]

    category_type = ['object', 'category', 'string', 'interval', 'bool']
    null_category_columns = [col for col in df.columns if (df[col].isnull().sum() > 0) and (df[col].dtype in category_type)]

    if len(null_numeric_columns) > 0:
        knn = KNNImputer()
        df[null_numeric_columns] = knn.fit_transform(df[null_numeric_columns])

    if len(null_category_columns) > 0:
        # label_encoder = LabelEncoder()
        # df[null_category_columns] = label_encoder.fit_transform(df[null_category_columns].astype(str))

        # df[null_category_columns] = df[null_category_columns].replace(label_encoder.transform(['nan'])[0], np.nan)

        # imputer = IterativeImputer(max_iter=10, random_state=0)
        # df[null_category_columns] = imputer.fit_transform(df[null_category_columns])
        # df[null_category_columns] = label_encoder.inverse_transform(df[null_category_columns].astype(int))

        label_encoders = {}

        for column in null_category_columns:
            label_encoder = LabelEncoder()
            df[column] = df[column].astype(str)
            df[column] = label_encoder.fit_transform(df[column].replace('nan', np.nan))
            label_encoders[column] = label_encoder

        imputer = IterativeImputer(max_iter=10, random_state=0)
        df[null_category_columns] = imputer.fit_transform(df[null_category_columns])
    
        print("herhere\n\n", df.isna().sum())

        for column in null_category_columns:
            label_encoder = label_encoders[column]
            df[column] = label_encoder.inverse_transform(df[column].astype(int))

    print(df.isna().sum())

    return 



# categorical label encoder

"""
 parameters :
    df: pandas dataframe accepted as data
    type: parameter to decide that which type of encoding is used (default value is onehot)
    columns: which columns of dataframe should be encoded
    sparse: to decide sparsity of data
"""


def one_hot_labelencoder(df, lable_encoding_type = 'onehot', columns = [], sparse = False):
    if isinstance(df, pd.DataFrame) == False:
        raise ValueError(f"Invalid datatype {type(df)} this function accept pandas dataframe")
    
    if len(columns) == 0:
        category_type = ['object', 'category', 'string', 'bool']
        columns = [col for col in df.columns if (df[col].dtype in category_type)]

    if len(columns) == 0:
        return 
    if lable_encoding_type == 'onehot':
        onehotencoder = OneHotEncoder(sparse_output=sparse)
        df[columns] = onehotencoder.fit_transform(df[columns])
    elif lable_encoding_type == 'labelencode':
        labelencoder = LabelEncoder()
        df[columns] = labelencoder.fit_transform(df[columns])
    else:
        raise ValueError("Invalid parameter did you mean 'onehot' or 'labelencode'?")




"""
    Trying pipeline module
"""


def get_data(df, keep='first', min_cat_percent = 5.0,
                datalosspercent = 10, min_var = 0.04, var_diff = 0.05,
                mod_diff = 0.05, numerical_column = [],
                categorical_column = [], temporal_column = [],
                temporal_type = 'interpolate', n_neighbors = 5,
                lable_encoding_type = 'onehot', columns = [], sparse = False):
    
    pipe = Pipeline([
        ('delete_duplicates', delete_duplicates(df)),
        ('create_category', make_category_columns(df)),
        ('drop_null', drop_row_column(df)),
        ('univariate_imputation', impute_row_column(df, numerical_column = [], categorical_column = [], temporal_column = [])),
        ('multivariate_imputation', multivariate_impute(df)),
        ('label_encoding' , one_hot_labelencoder(df,columns=[]))
    ])

    # data = pipe.fit_transform(df)
    return 0
    # return data

"""
    function ended
"""

def callingfunc():
    df = pd.read_csv('data\data_science_job.csv')

    # for col in df.columns:
    #     print(f"Column {col} :  {df[col].nunique()}")
    X = get_data(df.drop('target', axis=1))
    # print(X.shape)
callingfunc()