import pytest
import pandas as pd
from sklearn.base import BaseEstimator
import sys
sys.path.insert(0, '../src')
from VEDA import DimensionReducer

def test():
    df = pd.read_csv('tests\sample_data\sample.csv')
    X = df.drop(['price'], axis=1)
    y = df['price']

    dim_reducer = DimensionReducer.DimensionReducer()
    df, series = dim_reducer.fit_transform(X, y)
    
    # Assert the first output is a DataFrame
    assert isinstance(df, pd.DataFrame)
    
    # Assert the second output is a Series
    assert isinstance(series, pd.Series)