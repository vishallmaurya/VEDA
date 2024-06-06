import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
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
            
            print(new_df.shape)
            if change > datalosspercent:
                return df
            else:
                return new_df
            
def impute_row_column(df, limit = 0.04):
    null_count = df.isnull().sum().values.sum()
    if null_count == 0: # if data is completely clean
        return df
    else:
        limited_null_columns = [var for var in df.columns if df[var].isnull().mean() <= limit]
        excessive_null_columns = [var for var in df.columns if df[var].isnull().mean() > limit]
        
        if len(limited_null_columns) == 0:
            return df
        else:
            # checking for categorical column
            categorical_column = list(limited_null_columns[df.dtypes == 'object'])
            numerical_column = [col for col in limited_null_columns if col not in categorical_column]

            imputer1 = SimpleImputer(strategy='median')
imputer2 = SimpleImputer(strategy='mean')
def callingfunc():
    df = pd.read_csv('data\data_science_job.csv')
    print(df.shape)
    df = drop_row_column(df)
    print(df.shape)

callingfunc()