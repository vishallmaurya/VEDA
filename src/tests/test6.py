from VEDA import Veda
import pytest
import pandas as pd
from sklearn.base import BaseEstimator

def test():
    df = pd.read_csv('src\\tests\sample_data\sample.csv')
    X = df.drop(['price'], axis=1)
    y = df['price']

    veda = Veda.Veda(classification=False)
    df, series, outliers, strategy, model = veda.fit_transform(X, y)
    
    # Assert the first output is a DataFrame
    assert isinstance(df, pd.DataFrame)
    
    # Assert the second output is a Series
    assert isinstance(series, pd.Series)

    # Assert the third output is also a DataFrame
    assert isinstance(outliers, pd.DataFrame)

    # Assert the fourth output is a string
    assert isinstance(strategy, str)
    
    check = ["oversample", "combine", "none"]
        
    if strategy not in check:  # Condition matching the example
        assert isinstance(model, BaseEstimator)
    else:
        assert model is None