from VEDA import Preprocessor
import pytest
import pandas as pd

def test():
    df = pd.read_csv('src\\tests\sample_data\sample.csv')
    X = df.drop(['price'], axis=1)
    y = df['price']

    preprocess = Preprocessor.DataPreprocessor()
    df, series = preprocess.fit_transform(X, y)
    
    # Assert the first output is a DataFrame
    assert isinstance(df, pd.DataFrame)
    
    # Assert the second output is a Series
    assert isinstance(series, pd.Series)