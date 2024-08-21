# Licensed to the VISHAL MAURYA under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The VISHAL MAURYA licenses this file to You under
# the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# File: Veda.py
# Description: This file contains utility functions for performing complete EDA; it
# encompasses all functionalities of EDA.



from .BalanceData import AdaptiveBalancer
from .DimensionReducer import DimensionReducer
from .FeatureSelector import FeatureSelection
from .OutlierHandler import OutlierPreprocessor
from .Preprocessor import DataPreprocessor

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
        self.X, self.y, self.strategy, self.model = self.data_balancing.fit_transform(self.X, self.y)
        
        return self
        
    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
            raise ValueError("X must be a pandas DataFrame and y must be a pandas Series")
        
        if len(X) != len(y):
            raise RuntimeError("Size of X is not the same as y")

        return self.X, self.y, self.outliers, self.strategy, self.model
        
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
