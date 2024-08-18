import pytest
import pandas as pd
from sklearn.base import BaseEstimator
import sys
import os

# Adjust PYTHONPATH
sys.path.insert(0, 'src')

# Import BalanceData from VEDA
from VEDA import BalanceData

def test_example():
    # Use forward slashes for file paths
    csv_path = os.path.join('tests', 'sample_data', 'sample.csv')
    
    # Load data
    df = pd.read_csv(csv_path, encoding='ISO-8859-1')
    X = df.drop(['price'], axis=1)
    y = df['price']

    # Initialize the AdaptiveBalancer
    balancedata = BalanceData.AdaptiveBalancer(classification=False)
    df, series, strategy, model = balancedata.fit_transform(X, y)
    
    # Assertions
    assert isinstance(df, pd.DataFrame)
    assert isinstance(series, pd.Series)
    assert isinstance(strategy, str)
    
    check = ["oversample", "combine", "none"]
    if strategy not in check:
        assert isinstance(model, BaseEstimator)
    else:
        assert model is None
