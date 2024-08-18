import pytest
import pandas as pd
from sklearn.base import BaseEstimator
import sys
import os

# Adjust PYTHONPATH
sys.path.insert(0, 'src')

# Import Preprocessor from VEDA
from VEDA import Preprocessor

def test():
    # Use forward slashes for file paths
    csv_path = os.path.join('tests', 'sample_data', 'sample.csv')
    
    # Load data
    df = pd.read_csv(csv_path)

    X = df.drop(['price'], axis=1)
    y = df['price']

    # Initialize the DataPreprocessor
    preprocess = Preprocessor.DataPreprocessor()
    df_transformed, series_transformed = preprocess.fit_transform(X, y)
    
    # Assertions
    assert isinstance(df_transformed, pd.DataFrame)
    assert isinstance(series_transformed, pd.Series)
