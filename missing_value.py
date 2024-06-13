import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer

# CCA--> Complete case Analysis

"""
1. If there is no null values in dataframe then simply return 
2. It there is columns that have null values less than 4% then drop
   the rows from that columns
   
   2.1 If total dropped data is less than 10% then its fine else don't
   drop the rows return original data.
"""


def drop_row_column(df, datalosspercent = 10, limit = 0.04):
    null_count = df.isnull().sum().values.sum()
    if null_count == 0: # if data is completely clean
        return df
    else:
        limited_null_columns = [var for var in df.columns if df[var].isnull().mean() <= limit]
        excessive_null_columns = [var for var in df.columns if df[var].isnull().mean() > limit]

        if len(limited_null_columns) == 0:
            return df       
        else:
            new_df = df[limited_null_columns].dropna()
            new_df[excessive_null_columns] = df[excessive_null_columns]

            change = ((df.shape[0]-new_df.shape[0])/df.shape[0])*100

            if change > datalosspercent:
                return df
            else:
                return new_df

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

def impute_row_column(df, limit = 0.04, var_diff = 0.05, mod_diff = 0.05, numerical_column = [], categorical_column = [], temporal_column = [], temporal_type = 'interpolate'):
    
    """
        error handling
    """
    
    if(limit > 1 or limit < 0):
        raise ValueError("Invalid value. Value of limit should be from 0 to 1 inclusive.")
    if(var_diff > 1 or var_diff < 0):
        raise ValueError("Invalid value. Value of var_diff should be from 0 to 1 inclusive.")
    if(mod_diff > 1 or mod_diff < 0):
        raise ValueError("Invalid value. Value of mod_diff should be from 0 to 1 inclusive.")
    dtype_list = ['object', 'category', 'string', 'interval', 'bool', ]

    
    null_count = df.isnull().sum().values.sum()
    if null_count == 0: # if data is completely clean
        return df
    else:
        # columns that have less than given null values limit
        limited_null_columns = [var for var in df.columns if df[var].isnull().mean() <= limit]
        
        if len(limited_null_columns) == 0:
            return df
        else:
            # checking for categorical column
            
            if len(categorical_column) == 0:
                category_type = ['object', 'category', 'string', 'interval', 'bool']
                categorical_column = [col for col in df.columns if col in limited_null_columns and (df[col].dtype in category_type)]
            if len(categorical_column) == 0:
                numeric_type = ['int64', 'int32', 'int16', 'int8', 'uint64', 'uint32', 'uint16', 'uint8', 'float64', 'float32']
                numerical_column = [col for col in df.columns if col in limited_null_columns and (df[col].dtype in numeric_type)]
            if len(temporal_column) == 0:
                time_type = ['datetime64[ns]', 'timedelta64[ns]', 'datetime64[ns, tz]', 'period']
                temporal_column = [col for col in df.column if col in limited_null_columns and (df[col].dtype in time_type)]
            
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
                method_ = temporal_type + "()"
                new_data[temporal_column[i]] = df[temporal_column[i]].method_

            return new_data

# multivariate knn Imputer

def multivariate_impute(df, n_neighbors = 5):
    knn = KNNImputer()
    new_data = df
    new_data[df.columns] = knn.fit_transform(df)
    return new_data

def callingfunc():
    pass
    
callingfunc()