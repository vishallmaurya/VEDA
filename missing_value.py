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
            
            categorical_column = [col for col in df.columns if col in limited_null_columns and df[col].dtype == 'object']
            numerical_column = [col for col in limited_null_columns if col not in categorical_column]
            
            imputer1 = SimpleImputer(strategy='median')
            imputer2 = SimpleImputer(strategy='mean')
            imputer3 = SimpleImputer(strategy='constant',fill_value='Missing')
            imputer4 = SimpleImputer(strategy='most_frequent')

            trf = ColumnTransformer([
                ('imputer1',imputer1,numerical_column),
                # ('imputer2',imputer2,numerical_column)
            ],remainder='passthrough')

            new_data = df

            new_data[numerical_column] = trf.fit_transform(df[numerical_column])
            # print(new_data)
            # new_data = pd.DataFrame(new_data[:, 0], columns=numerical_column)
            # return new_data
            return new_data

def callingfunc():
    # df = pd.read_csv('data\data_science_job.csv')
    df = pd.read_csv('data\\titanic_toy.csv')
    # print(df.shape)
    # df = drop_row_column(df)
    # df.info()
    print(df.loc[df['Age'] == 22])
    df = impute_row_column(df.drop(['Survived', 'Family'], axis = 1), limit=0.06)
    print(df.loc[df['Age'] == 22])
callingfunc()