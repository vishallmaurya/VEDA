import pandas as pd
import numpy as np

# CCA--> Complete case Analysis

def drop_row_column(df):
    null_count = df.isnull().sum().values.sum()
    if null_count == 0: # if data is completely clean
        return df
    else:
        cols = [var for var in df.columns if df[var].isnull().mean() < 0.04]
        print(len(cols))
        if len(cols) == 0:
            return df       # there exists columns that have null values more than 5%, then it's not feasible to do cca
        else:
            new_df = df[cols].dropna() 
            return new_df


def callingfunc():
    df = pd.read_csv('data\data_science_job.csv')
    print(df.shape)
    df = drop_row_column(df)
    print(df.shape)

callingfunc()