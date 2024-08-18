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
# File: FeatureSelector.py
# Description: This file contains utility functions for efficiently selecting the 
# features that are important for the model.




from sklearn.base import TransformerMixin, BaseEstimator
import traceback
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, SelectKBest
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.utils.multiclass import type_of_target
from sklearn.base import BaseEstimator, TransformerMixin


class Standardizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        """Fits the scaler to the data."""
        # Check if X is a DataFrame or a 2D numpy array
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("X must be a pandas DataFrame or a 2D numpy ndarray.")
        
        # Ensure X is not empty
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must have at least one sample and one feature.")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(X)):
            raise ValueError("X contains NaN values, which cannot be processed.")
        if np.any(np.isinf(X)):
            raise ValueError("X contains infinite values, which cannot be processed.")
        
        self.scaler.fit(X)
        return self

    def transform(self, X, y=None):
        """Transforms the data by scaling features to zero mean and unit variance."""
        # Reuse the validation logic from fit
        self._validate_X(X)
        
        try:
            X_scaled = self.scaler.transform(X)
        except Exception as e:
            raise RuntimeError(f"An error occurred during transformation: {e}")
        
        return pd.DataFrame(X_scaled, columns=X.columns if isinstance(X, pd.DataFrame) else None)

    def fit_transform(self, X, y=None):
        """Fits the scaler and transforms the data in one step."""
        return self.fit(X, y).transform(X, y)
    
    def _validate_X(self, X):
        """Internal method to validate the input X."""
        # Check if X is a DataFrame or a 2D numpy array
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("X must be a pandas DataFrame or a 2D numpy ndarray.")
        
        # Ensure X is not empty
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must have at least one sample and one feature.")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(X)):
            raise ValueError("X contains NaN values, which cannot be processed.")
        if np.any(np.isinf(X)):
            raise ValueError("X contains infinite values, which cannot be processed.")
        


class CorrelationFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, percentile=90):
        """Initializes the CorrelationFeatureSelector with a given percentile."""
        if not (0 <= percentile <= 100):
            raise ValueError("Percentile must be between 0 and 100.")
        self.percentile = percentile
        self.selected_features_ = None

    def fit(self, X, y):
        """Fits the selector to the data by calculating correlations with the target variable."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series.")
        
        if len(X) != len(y):
            raise ValueError("The number of samples in X and y must be the same.")
        
        if X.empty:
            raise ValueError("X is empty. It must contain at least one feature.")
        
        if y.empty:
            raise ValueError("y is empty. It must contain at least one value.")
        
        correlations = X.corrwith(y).abs()

        if correlations.isna().all():
            raise ValueError("All correlation values are NaN.")
        if (correlations == 0).all():
            raise ValueError("All features have zero correlation with the target.")

        threshold = np.percentile(correlations, self.percentile)
        self.selected_features_ = correlations[correlations > threshold].index.tolist()

        return self

    def transform(self, X):
        """Transforms the data by selecting the features identified in the fit method."""
        if self.selected_features_ is None:
            raise RuntimeError("The fit method must be called before transform.")
        
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        
        if X.empty:
            raise ValueError("X is empty. It must contain at least one feature.")
        
        return self.selected_features_

    def fit_transform(self, X, y=None):
        """Fits the selector to the data and then transforms it."""
        return self.fit(X, y).transform(X)
    


class MutualInformationFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        """Initializes the selector with a given threshold for cumulative mutual information."""
        if not (0 < threshold <= 1):
            raise ValueError("Threshold must be between 0 and 1.")
        self.threshold = threshold
        self.selected_features_ = None
        self.optimal_k_ = None

    def _select_optimal_k(self, X, y):
        """Selects the optimal number of features based on mutual information."""
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("X must be a pandas DataFrame or a numpy ndarray.")
        
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError("y must be a pandas Series or a numpy ndarray.")
        
        if len(X) != len(y):
            raise ValueError("The number of samples in X and y must be the same.")

        X = np.array(X)
        y = np.array(y)
        
        if X.shape[1] == 0:
            raise ValueError("X must have at least one feature.")

        if X.size == 0 or y.size == 0:
            raise ValueError("X and y must not be empty.")

        target_type = type_of_target(y)

        try:
            if target_type == 'continuous':
                mi_scores = mutual_info_regression(X, y)
            else:
                mi_scores = mutual_info_classif(X, y)
        except Exception as e:
            raise RuntimeError(f"Failed to calculate mutual information scores: {e}")
        
        if len(mi_scores) != X.shape[1]:
            raise RuntimeError("Mismatch between number of features and mutual information scores computed.")

        sorted_indices = np.argsort(mi_scores)[::-1]
        sorted_mi_scores = mi_scores[sorted_indices]
        
        cumulative_mi_scores = np.cumsum(sorted_mi_scores)
        normalized_cumulative_mi_scores = cumulative_mi_scores / cumulative_mi_scores[-1]
        
        optimal_k = np.argmax(normalized_cumulative_mi_scores >= self.threshold) + 1
        optimal_k = min(optimal_k, X.shape[1])

        self.optimal_k_ = optimal_k
        return optimal_k

    def fit(self, X, y=None):
        """Fits the selector by calculating mutual information and selecting features."""
        if y is None:
            raise ValueError("y must be provided for mutual information calculation.")

        k = self._select_optimal_k(X, y)
        target_type = type_of_target(y)

        try:
            if target_type == 'continuous':
                self.selector_ = SelectKBest(mutual_info_regression, k=k)
            else:
                self.selector_ = SelectKBest(mutual_info_classif, k=k)
            self.selector_.fit(X, y)
        except Exception as e:
            raise RuntimeError(f"Failed to fit the mutual information selector: {e}")

        self.selected_features_ = self.selector_.get_support(indices=True)

        return self

    def transform(self, X):
        """Transforms the data by selecting the features identified during fitting."""
        if self.selected_features_ is None:
            raise RuntimeError("The fit method must be called before transform.")
        
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("X must be a pandas DataFrame or a numpy ndarray.")
        
        return self.selected_features_

    def fit_transform(self, X, y=None):
        """Fits the selector and then transforms the data."""
        return self.fit(X, y).transform(X)
    

class LassoFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, cv=5):
        """Initializes the Lasso feature selector with cross-validation."""
        if not isinstance(cv, int) or cv <= 0:
            raise ValueError("cv must be a positive integer.")
        self.cv = cv
        self.selected_features_ = None
        self.best_alpha_ = None

    def fit(self, X, y=None):
        """Fits the Lasso model and selects features based on non-zero coefficients."""
        if y is None:
            raise ValueError("y must be provided for Lasso regression.")
        
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("X must be a pandas DataFrame or a numpy ndarray.")
        
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError("y must be a pandas Series or a numpy ndarray.")
        
        if len(X) != len(y):
            raise ValueError("The number of samples in X and y must be the same.")

        X = np.array(X)
        y = np.array(y)
        
        if X.shape[1] == 0:
            raise ValueError("X must have at least one feature.")
        
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("X and y must not contain NaN values.")
        if np.any(np.isinf(X)) or np.any(np.isinf(y)):
            raise ValueError("X and y must not contain infinite values.")
        
        if y.ndim != 1:
            raise ValueError("y must be a 1D array or Series.")
        
        try:
            lasso_cv = LassoCV(cv=self.cv)
            lasso_cv.fit(X, y)
            
            self.best_alpha_ = lasso_cv.alpha_
            
            lasso = Lasso(alpha=self.best_alpha_)
            lasso.fit(X, y)
            
            self.selected_features_ = [i for i, coef in enumerate(lasso.coef_) if coef != 0]
            
        except ValueError as ve:
            raise ValueError(f"Value error occurred during Lasso regression: {ve}")
        except TypeError as te:
            raise TypeError(f"Type error occurred during Lasso regression: {te}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred during Lasso regression: {e}")
        
        return self

    def transform(self, X):
        """Transforms the data by selecting the features identified during fitting."""
        if self.selected_features_ is None:
            raise RuntimeError("The fit method must be called before transform.")
        
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("X must be a pandas DataFrame or a numpy ndarray.")
        
        return self.selected_features_

    def fit_transform(self, X, y=None):
        """Fits the selector and then transforms the data."""
        return self.fit(X, y).transform(X)
    

class AICBICFeatureSelector(BaseEstimator, TransformerMixin):
    """Transformer that selects features based on AIC and BIC criteria."""

    def __init__(self):
        self.best_aic_features_ = None
        self.best_bic_features_ = None
        self.feature_names_ = None

    def _validate_inputs(self, X, y):
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("X must be a pandas DataFrame or a numpy ndarray.")
        
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError("y must be a pandas Series or a numpy ndarray.")
        
        if len(X) != len(y):
            raise ValueError("The number of samples in X and y must be the same.")

        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        else:
            self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]

        return np.array(X), np.array(y)

    def _compute_aic_bic(self, X_subset, y):
        try:
            if isinstance(y, np.ndarray):
                y = pd.Series(y)

            X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.3, random_state=42)
            model = sm.OLS(y_train, sm.add_constant(X_train)).fit()
            predictions = model.predict(sm.add_constant(X_test))
            aic, bic = model.aic, model.bic
            r2 = r2_score(y_test, predictions)
            return aic, bic, r2
        
        except ValueError as ve:
            raise ValueError(f"Value error during model fitting or evaluation: {ve}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred during model fitting or evaluation: {e}")

    def fit(self, X, y):
        """Fit the transformer and select the best features based on AIC and BIC."""
        X, y = self._validate_inputs(X, y)

        best_aic_features = list(range(X.shape[1]))
        best_bic_features = best_aic_features.copy()

        try:
            best_aic, best_bic, _ = self._compute_aic_bic(X[:, best_aic_features], y)
        except Exception as e:
            raise RuntimeError(f"An error occurred while computing initial AIC/BIC: {e}")
        
        improved = True
        while improved:
            improved = False

            for feature in best_aic_features.copy():
                temp_aic_features = best_aic_features.copy()
                temp_aic_features.remove(feature)
                try:
                    current_aic, _, _ = self._compute_aic_bic(X[:, temp_aic_features], y)
                    if current_aic < best_aic:
                        best_aic = current_aic
                        best_aic_features = temp_aic_features
                        improved = True
                except Exception as e:
                    raise RuntimeError(f"An error occurred while computing AIC for feature subsets: {e}")

            for feature in best_bic_features.copy():
                temp_bic_features = best_bic_features.copy()
                temp_bic_features.remove(feature)
                try:
                    _, current_bic, _ = self._compute_aic_bic(X[:, temp_bic_features], y)
                    if current_bic < best_bic:
                        best_bic = current_bic
                        best_bic_features = temp_bic_features
                        improved = True
                except Exception as e:
                    raise RuntimeError(f"An error occurred while computing BIC for feature subsets: {e}")

        try:
            self.best_aic_features_ = [self.feature_names_[i] for i in best_aic_features]
            self.best_bic_features_ = [self.feature_names_[i] for i in best_bic_features]
        except IndexError as ie:
            raise IndexError(f"Index error when converting feature indices to names: {ie}")
        
        return self

    def transform(self, X):
        """Transform the data by selecting the features based on AIC and BIC criteria."""
        if self.best_aic_features_ is None or self.best_bic_features_ is None:
            raise RuntimeError("The transformer has not been fitted yet.")

        X = np.array(X) if isinstance(X, pd.DataFrame) else X

        return self.best_aic_features_, self.best_bic_features_
    
    def fit_transform(self, X, y=None):
        """Fits the selector and then transforms the data."""
        return self.fit(X, y).transform(X)
    

class FeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self, percentile=90, threshold=0.9, cv=5):
        self.percentile = percentile
        self.threshold = threshold
        self.cv = cv
    
        self.scaler = Standardizer()
        self.correlation_selector = CorrelationFeatureSelector(percentile=percentile)
        self.mi_selector = MutualInformationFeatureSelector(threshold=threshold)
        self.lasso_selector = LassoFeatureSelector(cv=cv)
        self.aic_bic_selector = AICBICFeatureSelector()

    def fit(self, X, y=None):
        """Fits all feature selection methods and the scaler."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series.")
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # Fit feature selectors
        self.correlation_selector.fit(X, y)
        self.mi_selector.fit(X_scaled, y)
        self.lasso_selector.fit(X_scaled, y)
        self.aic_bic_selector.fit(X, y)
        return self

    def transform(self, X, y=None):
        """Transforms X by selecting features based on various methods."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        
        X_scaled = self.scaler.transform(X)
        
        # Get selected features from each selector
        correlation_features = self.correlation_selector.transform(X)
        mi_features = self.mi_selector.transform(X_scaled)
        lasso_features = self.lasso_selector.transform(X_scaled)
        aic_features, bic_features = self.aic_bic_selector.transform(X)

        mi_features = [X.columns[i] for i in mi_features]
        lasso_features = [X.columns[i] for i in lasso_features]

        # Combine all selected features
        all_selected_features = list(set(correlation_features + 
                                         mi_features + 
                                         lasso_features + 
                                         aic_features +
                                         bic_features))

        X = X[all_selected_features]
        print(f"Selected features: {all_selected_features}")
        print(f"Shape of feature after feature selection:  {len(all_selected_features)}")
        return X, y
    
    def fit_transform(self, X, y=None):
        print(f"From feature selector file :   type of X : {type(X)}, and type of y:   {type(y)}")

        """Fits all selectors and scaler and then transforms X."""
        return self.fit(X, y).transform(X, y)
