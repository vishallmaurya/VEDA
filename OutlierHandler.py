import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from scipy.stats import shapiro, kstest, anderson, jarque_bera, skew, kurtosis
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import optuna
from diptest import diptest
import warnings

class OutlierHandlerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, tests=['skew-kurtosis'], method='default', handle='capping', minlen=5000, skew_thresh=1, kurt_thresh=1):
        # Validate that the tests parameter is a list
        if not isinstance(tests, list):
            raise TypeError(f"'tests' should be a list, but got {type(tests).__name__}.")
        
        # Ensure that method is a string
        if not isinstance(method, str):
            raise TypeError(f"'method' should be a string, but got {type(method).__name__}.")
        
        # Ensure that handle is a string
        if not isinstance(handle, str):
            raise TypeError(f"'handle' should be a string, but got {type(handle).__name__}.")

        if not isinstance(skew_thresh, (int, float)):
            raise ValueError("Parameter 'skew_thresh' should be a numerical value.")
        
        if not isinstance(kurt_thresh, (int, float)):
            raise ValueError("Parameter 'kurt_thresh' should be a numerical value.")
        
        # Validate minlen parameter
        if not isinstance(minlen, int) or minlen <= 0:
            raise ValueError("Parameter 'minlen' should be a positive integer.")
        
        self._validate_params()

        self.tests = tests
        self.method = method
        self.handle = handle
        self.minlen = minlen
        self.skew_thresh = skew_thresh
        self.kurt_thresh = kurt_thresh

        # Validate the parameters using the internal validation function

    def _validate_params(self):
        handle_list = ['capping', 'trimming', 'winsorization']
        if self.handle not in handle_list:
            raise ValueError(f"Invalid handle option. Expected one of {handle_list}, got {self.handle}")

        method_list = ['isolation-forest', 'lof', 'default']
        if self.method not in method_list:
            raise ValueError(f"Invalid method option. Expected one of {method_list}, got {self.method}")

        test_list = ['skew-kurtosis', 'shapiro', 'kstest', 'anderson', 'jarque-bera']
        for t in self.tests:
            if t not in test_list:
                raise ValueError(f"Invalid test '{t}'. Expected one of {test_list}.")

    def fit(self, X, y=None):
        return self  # Nothing to fit, but necessary for the Transformer interface

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"Expected X to be a pandas DataFrame, got {type(X)} instead.")
        if y is not None and not isinstance(y, pd.Series):
            raise ValueError(f"Expected y to be a pandas Series, got {type(y)} instead.")

        # Handling outliers
        outliers, cleaned_x, cleaned_y = self._handle_outliers(X, y)

        return outlers, cleaned_x, cleaned_y

    def _handle_outliers(self, data, y=None):
        numerical_type = ['int64', 'int32', 'int16', 'int8', 'uint64', 'uint32', 'uint16', 'uint8', 'float64', 'float32']

        for col in data.columns:
            if data[col].dtype not in numerical_type:
                raise ValueError(f"Invalid dtype in column '{col}'. Expected one of {numerical_type}.")

        # Prepare to track outliers
        outlier_indices = []

        # 1. Check for multimodal columns using Dip Test
        multimodal_columns = []
        for column in data.columns:
            warnings.simplefilter("ignore")
            dip_stat, dip_p_value = diptest(data[column])
            if dip_p_value < 0.05:
                multimodal_columns.append(column)

        if multimodal_columns:
            print(f"Multimodal distribution detected in columns: {multimodal_columns}")
            for column in multimodal_columns:
                sampled_data = self._stratified_sample(data, column, n_samples=int(0.15 * len(data)), n_splits=5)
                dbscan = DBSCAN()
                labels = dbscan.fit_predict(sampled_data[[column]])
                noise = sampled_data[labels == -1]
                outlier_indices.extend(noise.index)

        if not outlier_indices:
            # 2. Apply Isolation Forest
            if (self.method == 'default' and data.shape[0] >= 10000) or (self.method == 'isolation-forest'):
                print("Isolation Forest is used for outlier detection.")
                iso_forest = self._tune_isolation_forest_params_with_optuna(data)
                outlier = iso_forest.fit_predict(data)
                outlier_indices.extend(data.index[outlier == -1])

            # 3. Apply LOF
            if not outlier_indices:
                k = int(sqrt(len(data)))
                variance_threshold = 0.05
                dip_p_value_threshold = 0.05

                nbrs = NearestNeighbors(n_neighbors=k).fit(data)
                distances, _ = nbrs.kneighbors(data)
                distances = distances[:, k - 1]

                dist_variance = np.var(distances)
                dip, dip_p_value = diptest(distances)

                use_lof = (dist_variance > variance_threshold) and (dip_p_value < dip_p_value_threshold)

                if use_lof or self.method == 'lof':
                    print("Local Outlier Factor (LOF) is used for outlier detection.")
                    lof = LocalOutlierFactor(n_neighbors=k)
                    y_pred = lof.fit_predict(data)
                    outlier_indices.extend(data.index[y_pred == -1])

            # 4. Apply the normal distribution method
            if not outlier_indices and self.method == 'default':
                print("Default method for outlier detection.")
                for column in data.columns:
                    if self._is_normal_distribution(data[column]):
                        mean = data[column].mean()
                        std = data[column].std()
                        upper_limit = mean + 3 * std
                        lower_limit = mean - 3 * std
                        outliers = (data[column] > upper_limit) | (data[column] < lower_limit)
                        outlier_indices.extend(data.index[outliers])

                        if self.handle == 'capping':
                            data[column] = np.where(data[column] > upper_limit, upper_limit,
                                                    np.where(data[column] < lower_limit, lower_limit, data[column]))
                        elif self.handle == 'trimming':
                            data = data[~outliers]
                            y = y[~outliers]
                        elif self.handle == 'winsorization':
                            data[column] = data[column].clip(lower=lower_limit, upper=upper_limit)

        # Remove duplicate indices and keep only valid indices
        outlier_indices = list(set(outlier_indices))
        outlier_indices = [i for i in outlier_indices if 0 <= i < len(data)]

        if outlier_indices:
            valid_indices = [idx for idx in outlier_indices if idx in data.index and (y is None or idx in y.index)]
            outliers = data.loc[outlier_indices]
            cleaned_data = data.drop(valid_indices)
            cleaned_y = y.drop(valid_indices) if y is not None else None
        else:
            outliers = pd.DataFrame(columns=data.columns)
            cleaned_data = data
            cleaned_y = y

        return outliers, cleaned_data, cleaned_y

    def _stratified_sample(self, data, column, n_samples, n_splits=5):
        data['quantile_bin'] = pd.qcut(data[column], q=n_splits, labels=False, duplicates='drop')
        stratified_data, _ = train_test_split(data, stratify=data['quantile_bin'], 
                                              train_size=n_samples, random_state=42)
        stratified_data = stratified_data.drop(columns=['quantile_bin'])
        return stratified_data

    def is_normal_distribution(self, data):
        """
        Check if the data follows a normal distribution based on selected statistical tests.

        Parameters:
        - data: pandas Series or array-like, the data to test.

        Returns:
        - bool: True if data is likely normal, False otherwise.
        """
        if not isinstance(data, pd.Series):
            data = pd.Series(data)

        numerical_type = ['int64', 'int32', 'int16', 'int8', 'uint64', 'uint32', 'uint16', 'uint8', 'float64', 'float32']
        if data.dtype not in numerical_type:
            raise ValueError(f"Data should contain only numerical values; permitted data types are {numerical_type}")
        
        # Perform Shapiro test if the data length is sufficient
        if (len(data) >= self.minlen) and ('shapiro' in self.tests):
            shapiro_stat, shapiro_p_value = shapiro(data)
            if shapiro_p_value <= 0.05:
                return False

        # Check skewness and kurtosis
        if 'skew-kurtosis' in self.tests:
            data_skewness = skew(data)
            data_kurtosis = kurtosis(data, fisher=False)
            if abs(data_skewness) > self.skew_thresh or abs(data_kurtosis - 3) > self.kurt_thresh:
                return False

        # Kolmogorov-Smirnov test
        if 'kstest' in self.tests:
            ks_stat, ks_p_value = kstest(data, 'norm', args=(data.mean(), data.std()))
            if ks_p_value <= 0.05:
                return False

        # Anderson-Darling test
        if 'anderson' in self.tests:
            anderson_result = anderson(data)
            if any(anderson_result.statistic >= cv for cv in anderson_result.critical_values):
                return False

        # Jarque-Bera test
        if 'jarque-bera' in self.tests:
            jb_stat, jb_p_value = jarque_bera(data)
            if jb_p_value <= 0.05:
                return False

        return True

    def _tune_isolation_forest_params_with_optuna(self, X):
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self._objective(trial, X), n_trials=self._calculate_optimal_trials(len(X)))

        best_params = study.best_params
        best_estimator = IsolationForest(**best_params, random_state=42)
        return best_estimator

    def _objective(self, trial, X):
        contamination = trial.suggest_float('contamination', 0.01, 0.1)
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_samples = trial.suggest_int('max_samples', 1, len(X))
        random_state = 42

        X_train, X_val = train_test_split(X, test_size=0.2, random_state=random_state)

        iso_forest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state
        )
        iso_forest.fit(X_train)

        y_pred = iso_forest.predict(X_val)
        y_true = np.ones_like(y_pred)
        y_true[y_pred == -1] = -1  

        return f1_score(y_true, y_pred)

    def _calculate_optimal_trials(self, dataset_size):
        if dataset_size < 1000:
            return 100
        elif dataset_size < 10000:
            return 65
        elif dataset_size < 100000:
            return 40
        else:
            return 20

    def processOutlier(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"X should be a pandas DataFrame, got {type(X)} instead.")
        if not isinstance(y, pd.Series):
            raise ValueError(f"y should be a pandas Series, got {type(y)} instead.")

        try:
            outliers, cleaned_x, cleaned_y = self.transform(X, y)
        except Exception as e:
            raise RuntimeError(f"Error during outlier handling: {str(e)}")

        print("data size: ", X.shape)
        print("y size and type: ", y.size, " and ", type(y))
        print("outlier size: ", outliers.shape)
        print("cleaned_x size: ", cleaned_x.shape)
        print("cleaned_y size: ", cleaned_y.shape)
        print("ok after getting :  ", type(cleaned_y))

        return outliers, cleaned_x, cleaned_y