import pytest
import pandas as pd
from sklearn.base import BaseEstimator
import sys
import os

# Adjust PYTHONPATH
sys.path.insert(0, 'src')

# Import OutlierHandler from VEDA
from VEDA import OutlierHandler

def test():
    # Use forward slashes for file paths
    csv_path = os.path.join('tests', 'sample_data', 'sample.csv')
    
    # Load data
    df = pd.read_csv(csv_path)

    X = df.drop(['price'], axis=1)
    y = df['price']

    # Initialize the OutlierPreprocessor
    outlier = OutlierHandler.OutlierPreprocessor()
    df_transformed, series_transformed, outliers = outlier.fit_transform(X, y)
    
    # Assertions
    assert isinstance(df_transformed, pd.DataFrame)
    assert isinstance(series_transformed, pd.Series)
    assert isinstance(outliers, pd.DataFrame)
