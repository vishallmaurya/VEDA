import pytest
import pandas as pd
from sklearn.base import BaseEstimator
import sys
import os

# Adjust PYTHONPATH
sys.path.insert(0, 'src')

# Import Veda from VEDA
from veda_lib import Veda

def test():
    # Use forward slashes for file paths
    csv_path = os.path.join('tests', 'sample_data', 'sample.csv')
    
    # Load data
    df = pd.read_csv(csv_path)
    X = df.drop(['TenYearCHD'], axis=1)
    y = df['TenYearCHD']

    # Initialize Veda
    veda = Veda.Veda(classification=True)
    df_transformed, series_transformed, outliers, strategy, model = veda.fit_transform(X, y)
    
    # Assertions
    assert isinstance(df_transformed, pd.DataFrame)
    assert isinstance(series_transformed, pd.Series)
    assert isinstance(outliers, pd.DataFrame)
    assert isinstance(strategy, str)
    
    # Check if strategy is one of the expected values
    check = ["oversample", "combine", "none"]
    if strategy not in check:
        assert isinstance(model, BaseEstimator)
    else:
        assert model is None
