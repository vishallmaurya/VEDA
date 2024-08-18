from BalanceData import AdaptiveBalancer
from DimensionReducer import DimensionReducer
from FeatureSelector import FeatureSelection
from OutlierHandler import OutlierPreprocessor
from Preprocessor import DataPreprocessor

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class Veda(BaseEstimator, TransformerMixin):
    def __init__(self, classification=None):
        if classification is None:
            raise ValueError("Parameter classification can't be empty")
        self.preprocess_data = DataPreprocessor()
        self.outlierhandler = OutlierPreprocessor()
        self.feature_selection = FeatureSelection()
        self.reduced_dim = DimensionReducer()
        self.data_balancing = AdaptiveBalancer(classification=classification)
        self.X = None
        self.y = None
        self.outliers = None
        self.strategy = None
        self.model = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
            raise ValueError("X must be a pandas DataFrame and y must be a pandas Series")
        
        if len(X) != len(y):
            raise RuntimeError("Size of X is not the same as y")
        
        self.X, self.y = self.preprocess_data.fit(X, y).transform(X, y)
        self.X, self.y, self.outliers = self.outlierhandler.fit_transform(self.X, self.y)
        self.X, self.y = self.feature_selection.fit_transform(self.X, self.y)
        self.X, self.y = self.reduced_dim.fit_transform(self.X, self.y)
        print("size before balancing:  ", self.X.shape, " and  ", self.y.shape )
        self.X, self.y, self.strategy, self.model = self.data_balancing.fit_transform(self.X, self.y)
        print("size after balancing:  ", self.X.shape, " and  ", self.y.shape )

        return self
        
    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
            raise ValueError("X must be a pandas DataFrame and y must be a pandas Series")
        
        if len(X) != len(y):
            raise RuntimeError("Size of X is not the same as y")

        return self.X, self.y, self.outliers, self.strategy, self.model
        
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
