import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN, SMOTEN
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from collections import Counter
from sklearn.cluster import DBSCAN
from scipy.stats import entropy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError


class AdaptiveBalancer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.5, classification=None):
        if classification is None:
            raise ValueError("Parameter classification can't be empty")
        if not isinstance(threshold, (int, float)):
            raise ValueError("Parameter threshold must be a number")
        self.threshold = threshold
        self.classification = classification
        self.strategy = None
        self.sampler = None
        self.model = None

    def _calculate_dynamic_threshold(self, y):
        try:
            class_ratios = np.array(list(Counter(y).values())) / len(y)
            iqr = np.percentile(class_ratios, 75) - np.percentile(class_ratios, 25)
            dynamic_threshold = np.median(class_ratios) - (iqr / 2)
            return max(0.1, dynamic_threshold)
        except Exception as e:
            raise RuntimeError("Error calculating dynamic threshold.") from e

    def _calculate_entropy(self, y):
        try:
            class_probs = np.array(list(Counter(y).values())) / len(y)
            return entropy(class_probs, base=2)
        except Exception as e:
            raise RuntimeError("Error calculating entropy.") from e

    def _estimate_class_density(self, X, y):
        try:
            db = DBSCAN().fit(X)
            labels = db.labels_

            majority_density = np.sum(labels == 0) / len(y[y == 0]) if len(y[y == 0]) > 0 else 1
            minority_density = np.sum(labels == 1) / len(y[y == 1]) if len(y[y == 1]) > 0 else 1

            density_ratio = (minority_density / majority_density) if majority_density > 0 else 1

            return density_ratio
        except Exception as e:
            raise RuntimeError("Error estimating class density.") from e

    def _adaptive_threshold(self, X, y):
        try:
            n_samples = len(y)
            class_ratios = np.array(list(Counter(y).values())) / n_samples
            small_size = np.percentile(class_ratios, 10) * n_samples
            large_size = np.percentile(class_ratios, 90) * n_samples
            imbalance_threshold = np.percentile(class_ratios, 25)

            return small_size, large_size, imbalance_threshold
        except Exception as e:
            raise RuntimeError("Error calculating adaptive thresholds.") from e

    def _select_balancing_strategy(self, X, y):
        try:
            small_size, large_size, imbalance_threshold = self._adaptive_threshold(X, y)
            counter = Counter(y)
            majority_class = max(counter, key=counter.get)
            minority_class = min(counter, key=counter.get)
            majority_count = counter[majority_class]
            minority_count = counter[minority_class]
            imbalance_ratio = minority_count / majority_count

            dynamic_threshold = self._calculate_dynamic_threshold(y)
            entropy_value = self._calculate_entropy(y)
            density_ratio = self._estimate_class_density(X, y)

            if imbalance_ratio >= max(self.threshold, dynamic_threshold):
                self.strategy = "none"
                self.sampler = None
            elif density_ratio < 0.5:
                self.strategy = "oversample"
                self.sampler = SMOTEN() if entropy_value > 0.5 else SMOTE()
            elif len(X) < small_size:
                self.strategy = "oversample"
                self.sampler = ADASYN() if imbalance_ratio > 0.2 else SMOTE()
            elif len(X) > large_size:
                self.strategy = "combine"
                self.sampler = SMOTEENN() if imbalance_ratio > imbalance_threshold else SMOTETomek()
            elif imbalance_ratio < imbalance_threshold:
                self.strategy = "anomaly"
                self.sampler = IsolationForest(contamination=imbalance_ratio)
            else:
                self.strategy = "ensemble"
                self.sampler = RandomForestClassifier(class_weight='balanced_subsample')

        except Exception as e:
            raise RuntimeError("Error selecting balancing strategy.") from e

    def fit(self, X, y=None):
        if not self.classification:
            return self

        if y is None:
            raise ValueError("Target variable 'y' cannot be None when classification is set to True.")
        
        self._select_balancing_strategy(X, y)
        
        if self.strategy == "none":
            print("Data is already balanced or within the acceptable threshold.")
        elif self.strategy == "anomaly":
            self.model = self.sampler.fit(X)
        elif self.strategy == "ensemble":
            self.model = self.sampler.fit(X, y)
        
        return self

    def transform(self, X, y=None):
        if not self.classification:
            return X, y

        if y is None:
            raise ValueError("Target variable 'y' cannot be None when classification is set to True.")
        
        if self.strategy == "none":
            return X, y, self.strategy, None
        print(self.strategy, " is used")

        if self.strategy in ["oversample", "combine"]:
            X_res, y_res = self.sampler.fit_resample(X, y)
            return X_res, y_res, self.strategy, None
        elif self.strategy == "anomaly":
            if self.model is None:
                raise NotFittedError("The model has not been fitted yet. Please fit the model before transforming the data.")
            y_res = self.model.predict(X)
            return X, y_res, self.strategy, self.model
        elif self.strategy == "ensemble":
            if self.model is None:
                raise NotFittedError("The model has not been fitted yet. Please fit the model before transforming the data.")
            return X, y, self.strategy, self.model

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
