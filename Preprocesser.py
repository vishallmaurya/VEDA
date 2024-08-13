import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline,make_pipeline

from sklearn.base import BaseEstimator, TransformerMixin

"""
    parameter:
        df: pandas dataframe
"""

class DeleteDuplicates(BaseEstimator, TransformerMixin):
    def __init__(self, keep='first'):
        if keep not in ['first', 'last', False]:
            raise ValueError("Invalid value for 'keep'. Must be 'first', 'last', or False.")
        self.keep = keep

    def _delete_duplicates(self, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, but got {type(df).__name__}.")
        
        try:
            df.drop_duplicates(keep=self.keep, inplace=True)
        except Exception as e:
            raise RuntimeError(f"An error occurred while dropping duplicates: {str(e)}")
        
        return df
    
    def fit(self, df, y=None):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, but got {type(df).__name__}.")
        return self
    
    def transform(self, df, y=None):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, but got {type(df).__name__}.")
        
        try:
            df = self._delete_duplicates(df)
        except Exception as e:
            raise RuntimeError(f"An error occurred during transformation: {str(e)}")
        
        return df


class MakeCategoryColumns(BaseEstimator, TransformerMixin):
    def __init__(self, min_cat_percent=5.0):
        if not (0 <= min_cat_percent <= 100):
            raise ValueError("min_cat_percent should be between 0 and 100.")
        self.min_cat_percent = min_cat_percent

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, but got {type(X).__name__}.")
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, but got {type(X).__name__}.")
        
        try:
            df = X.copy()
            for col in df.columns:
                if pd.api.types.is_object_dtype(df[col]) == False:
                    count = df[col].nunique()
                    percent_count = (count / df[col].shape[0]) * 100 

                    if percent_count <= self.min_cat_percent:
                        df[col] = df[col].astype('object')
            return df
        except Exception as e:
            raise RuntimeError(f"An error occurred during transformation: {str(e)}")

 
class DropRowColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, datalosspercent=10, min_var=0.4):
        if not (0 <= datalosspercent <= 100):
            raise ValueError("datalosspercent should be between 0 and 100.")
        if not (0 <= min_var <= 1):
            raise ValueError("min_var should be between 0 and 1.")
        
        self.datalosspercent = datalosspercent
        self.min_var = min_var

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"Expected a pandas DataFrame, but got {type(X).__name__}.")
        
        self._validate_dataframe(X)
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"Expected a pandas DataFrame, but got {type(X).__name__}.")

        self._validate_dataframe(X)
        null_count = X.isnull().sum().values.sum()
        
        for col in X.columns:
            if X[col].nunique() == X.shape[0]:
                X.drop(col, axis=1, inplace=True)

        if null_count == 0:
            return X
        
        limited_null_columns = [var for var in X.columns if X[var].isnull().mean() <= self.min_var]
        
        if len(limited_null_columns) == 0:
            return X

        new_df = X[limited_null_columns].dropna()
        excessive_null_columns = [var for var in X.columns if X[var].isnull().mean() > self.min_var]
        new_df[excessive_null_columns] = X[excessive_null_columns]
        
        change = ((X.shape[0] - new_df.shape[0]) / X.shape[0]) * 100
        
        if change > self.datalosspercent:
            return X
        else:
            return new_df

    def _validate_dataframe(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Expected a pandas DataFrame, but got {type(df).__name__}.")
        if df.empty:
            raise ValueError("DataFrame is empty.")


class ImputeRowColumn(BaseEstimator, TransformerMixin):
    def __init__(self, min_var = 0.04, var_diff = 0.05, mod_diff = 0.05, numerical_column = None, categorical_column = None, temporal_column = None, temporal_type = 'interpolate'):
        if not (0 <= min_var <= 1):
            raise ValueError("min_var should be between 0 and 1.")
        if not (0 <= var_diff <= 1):
            raise ValueError("var_diff should be between 0 and 1")
        if not (0 <= mod_diff <= 1):
            raise ValueError("mod_diff should be between 0 and 1")

        self.min_var = min_var
        self.var_diff = var_diff
        self.mod_diff = mod_diff
        self.numerical_column = numerical_column if numerical_column is not None else []
        self.categorical_column = categorical_column if categorical_column is not None else []
        self.temporal_column = temporal_column if temporal_column is not None else []
        self.temporal_type = temporal_type

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"Expected a pandas DataFrame, but got {type(X).__name__}.")
        
        self._validate_dataframe(X)
        return self

    def transform(self, X):
        df = X.copy()
        
        # Error handling
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Invalid datatype {type(df)}. This function accepts a pandas DataFrame.")

        dtype_list = [
            'object', 'category', 'string', 'interval', 'bool',
            'int64', 'int32', 'int16', 'int8', 'uint64', 'uint32',
            'uint16', 'uint8', 'float64', 'float32',
            'datetime64[ns]', 'timedelta64[ns]', 'datetime64[ns, tz]', 'period'
        ]

        for col in df.columns:
            if df[col].dtype not in dtype_list:
                raise ValueError(f"Invalid dtype in column '{col}'. Supported dtypes: {dtype_list}")

        null_count = df.isnull().sum().values.sum()
        if null_count == 0:
            return df

        limited_null_columns = [col for col in df.columns if df[col].isnull().mean() <= self.min_var]
        if not limited_null_columns:
            return df
        
        # Determine columns if not provided
        if not self.categorical_column:
            category_type = ['object', 'category', 'string', 'interval', 'bool']
            self.categorical_column = [col for col in limited_null_columns if df[col].dtype in category_type]
        
        if not self.numerical_column:
            numeric_type = ['int64', 'int32', 'int16', 'int8', 'uint64', 'uint32', 'uint16', 'uint8', 'float64', 'float32']
            self.numerical_column = [col for col in limited_null_columns if df[col].dtype in numeric_type]
        
        if not self.temporal_column:
            time_type = ['datetime64[ns]', 'timedelta64[ns]', 'datetime64[ns, tz]', 'period']
            self.temporal_column = [col for col in limited_null_columns if df[col].dtype in time_type]

        imputer_median = SimpleImputer(strategy='median')
        imputer_mean = SimpleImputer(strategy='mean')
        imputer_mode = SimpleImputer(strategy='most_frequent')

        # Transformations
        trf_median = ColumnTransformer([('imputer_median', imputer_median, self.numerical_column)], remainder='passthrough')
        trf_mean = ColumnTransformer([('imputer_mean', imputer_mean, self.numerical_column)], remainder='passthrough')
        trf_mode = ColumnTransformer([('imputer_mode', imputer_mode, self.categorical_column)], remainder='passthrough')

        new_data = df.copy()
        variation_before_imputation = df[self.numerical_column].var()

        median = trf_median.fit_transform(new_data[self.numerical_column])
        mean = trf_mean.fit_transform(new_data[self.numerical_column])
        mode = trf_mode.fit_transform(new_data[self.categorical_column])

        # Handling numerical columns
        for i, col in enumerate(self.numerical_column):
            percentage_change_median = abs((variation_before_imputation.iloc[i] - pd.Series(median[:, i]).var()) / variation_before_imputation.iloc[i])
            percentage_change_mean = abs((variation_before_imputation.iloc[i] - pd.Series(mean[:, i]).var()) / variation_before_imputation.iloc[i])

            if (percentage_change_median > percentage_change_mean) and (percentage_change_mean < self.var_diff):
                new_data[col] = mean[:, i]
            elif percentage_change_median < self.var_diff:
                new_data[col] = median[:, i]

        # Handling categorical columns
        for i, col in enumerate(self.categorical_column):
            values = df[col].value_counts().sort_values(ascending=False)
            if len(values) > 1 and values.iloc[1] / values.iloc[0] > self.mod_diff:
                continue
            else:
                new_data[col] = mode[:, i]

        # Handling temporal columns
        for col in self.temporal_column:
            if self.temporal_type == 'bfill':
                new_data[col] = df[col].bfill()
            elif self.temporal_type == 'ffill':
                new_data[col] = df[col].ffill()
            elif self.temporal_type == 'interpolate':
                new_data[col] = df[col].interpolate()
            else:
                raise ValueError("Incorrect temporal_type method used")

        return new_data

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def _validate_dataframe(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Expected a pandas DataFrame, but got {type(df).__name__}.")
        if df.empty:
            raise ValueError("DataFrame is empty.")


class MultivariateImputer(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=5):
        """
        Parameters:
        - n_neighbors: Number of neighbors for the KNN imputer. Default is 5.
        """
        if not isinstance(n_neighbors, int) or n_neighbors <= 0:
            raise ValueError("n_neighbors must be a positive integer.")
        
        self.n_neighbors = n_neighbors
        self.label_encoders = {}

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"Expected a pandas DataFrame, but got {type(X).__name__}.")
        
        self._validate_dataframe(X)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"Expected a pandas DataFrame, but got {type(X).__name__}.")
        
        self._validate_dataframe(X)

        numeric_types = ['int64', 'int32', 'int16', 'int8', 'uint64', 'uint32', 'uint16', 'uint8', 'float64', 'float32']
        null_numeric_columns = [col for col in X.columns if (X[col].isnull().sum() > 0) and (X[col].dtype in numeric_types)]

        category_types = ['object', 'category', 'string', 'interval', 'bool']
        null_category_columns = [col for col in X.columns if (X[col].isnull().sum() > 0) and (X[col].dtype in category_types)]

        X.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Impute numeric columns using KNNImputer
        if len(null_numeric_columns) > 0:
            knn_imputer = KNNImputer(n_neighbors=self.n_neighbors)
            X[null_numeric_columns] = knn_imputer.fit_transform(X[null_numeric_columns])

        # Impute categorical columns using LabelEncoder and IterativeImputer
        if len(null_category_columns) > 0:
            self.label_encoders = {}
            
            # Label encode and handle missing values
            for column in null_category_columns:
                X[column] = X[column].fillna('Unknown')
                label_encoder = LabelEncoder()
                X[column] = label_encoder.fit_transform(X[column])
                self.label_encoders[column] = label_encoder

            # Replace 'Unknown' with np.nan for imputation
            for column in null_category_columns:
                code = self.label_encoders[column].transform(['Unknown'])[0]
                X[column] = X[column].replace(code, np.nan)
            
            imputer = IterativeImputer(max_iter=10, random_state=42)
            X[null_category_columns] = imputer.fit_transform(X[null_category_columns])

            # Convert back to original labels
            for column in null_category_columns:
                X[column] = X[column].round().astype(int)
                X[column] = X[column].apply(lambda x: x if x < len(self.label_encoders[column].classes_) else len(self.label_encoders[column].classes_) - 1)
                X[column] = self.label_encoders[column].inverse_transform(X[column])
       
        return X

    def _validate_dataframe(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Expected a pandas DataFrame, but got {type(df).__name__}.")
        if df.empty:
            raise ValueError("DataFrame is empty.")



class OneHotLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, label_encoding_type='default', min_category=5, columns=None, sparse=False):
        """
        Parameters:
        - label_encoding_type: Type of encoding ('default', 'onehot', or 'labelencode').
        - min_category: Minimum number of unique categories to switch from label encoding to one-hot encoding.
        - columns: List of columns to encode. If None, automatically selects categorical columns.
        - sparse: If True, return sparse matrix from one-hot encoding.
        """
        if label_encoding_type not in ['default', 'onehot', 'labelencode']:
            raise ValueError("Invalid label_encoding_type. Expected 'default', 'onehot', or 'labelencode'.")

        if not isinstance(min_category, int) or min_category <= 0:
            raise ValueError("min_category must be a positive integer.")

        if columns is not None and not isinstance(columns, list):
            raise ValueError("columns should be a list of column names or None.")

        if not isinstance(sparse, bool):
            raise ValueError("sparse should be a boolean value.")

        self.label_encoding_type = label_encoding_type
        self.min_category = min_category
        self.columns = columns
        self.sparse = sparse
        self.onehot_encoders = {}
        self.label_encoders = {}

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"Expected a pandas DataFrame, but got {type(X).__name__}.")
        
        self._select_columns(X)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"Expected a pandas DataFrame, but got {type(X).__name__}.")
        
        self._select_columns(X)

        for col in self.columns:
            number_of_category = X[col].nunique()

            if self.label_encoding_type == 'onehot' or (self.label_encoding_type == 'default' and number_of_category <= self.min_category):
                onehotencoder = OneHotEncoder(sparse_output=self.sparse)
                transformed_data = onehotencoder.fit_transform(X[[col]])
                new_columns = onehotencoder.get_feature_names_out([col])
                transformed_df = pd.DataFrame(transformed_data.toarray(), columns=new_columns, index=X.index)
                X = pd.concat([X, transformed_df], axis=1)
                X.drop(columns=col, inplace=True)
                self.onehot_encoders[col] = onehotencoder

            elif self.label_encoding_type == 'default' or self.label_encoding_type == 'labelencode':
                labelencoder = LabelEncoder()
                X[col] = labelencoder.fit_transform(X[col])
                self.label_encoders[col] = labelencoder

            else:
                raise ValueError("Invalid label_encoding_type. Expected 'onehot' or 'labelencode'.")
        
        return X

    def _select_columns(self, X):
        if self.columns is None:
            category_type = ['object', 'category', 'string', 'bool']
            self.columns = [col for col in X.columns if X[col].dtype in category_type]
        
        if len(self.columns) == 0:
            raise ValueError("No categorical columns found for encoding.")
    
    def inverse_transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"Expected a pandas DataFrame, but got {type(X).__name__}.")
        
        for col in self.columns:
            if col in self.label_encoders:
                X[col] = self.label_encoders[col].inverse_transform(X[col])
            elif col in self.onehot_encoders:
                onehot_columns = self.onehot_encoders[col].get_feature_names_out([col])
                X[col] = self.onehot_encoders[col].inverse_transform(X[onehot_columns])
                X.drop(columns=onehot_columns, inplace=True)
        
        return X


class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, keep='first', min_cat_percent=5.0, datalosspercent=10, 
                 min_var=0.04, var_diff=0.05, mod_diff=0.05, numerical_column=None,
                 categorical_column=None, temporal_column=None, temporal_type='interpolate', 
                 n_neighbors=5, label_encoding_type='onehot', columns=None, sparse=False):
        
        self.keep = keep
        self.min_cat_percent = min_cat_percent
        self.datalosspercent = datalosspercent
        self.min_var = min_var
        self.var_diff = var_diff
        self.mod_diff = mod_diff
        self.numerical_column = numerical_column or []
        self.categorical_column = categorical_column or []
        self.temporal_column = temporal_column or []
        self.temporal_type = temporal_type
        self.n_neighbors = n_neighbors
        self.label_encoding_type = label_encoding_type
        self.columns = columns or []
        self.sparse = sparse

        # Initialize the pipeline
        self.pipeline = Pipeline([
            ('delete_duplicates', DeleteDuplicates(keep=self.keep)),
            ('create_category', MakeCategoryColumns(min_cat_percent=self.min_cat_percent)),
            ('drop_null', DropRowColumnTransformer(datalosspercent=self.datalosspercent, min_var=self.min_var)),
            ('univariate_imputation', ImputeRowColumn(numerical_column=self.numerical_column,
                                                                      categorical_column=self.categorical_column,
                                                                      temporal_column=self.temporal_column,
                                                                      temporal_type=self.temporal_type)),
            ('multivariate_imputation', MultivariateImputer(n_neighbors=self.n_neighbors)),
            ('label_encoding', OneHotLabelEncoder(label_encoding_type=self.label_encoding_type, 
                                                             columns=self.columns, sparse=self.sparse)),
        ])

    def fit(self, X, y=None):
        self.pipeline.fit(X, y)
        return self

    def transform(self, X):
        try:
            return self.pipeline.transform(X)
        except Exception as e:
            raise RuntimeError(f"Error occurred during pipeline transformation: {str(e)}")

    def fit_transform(self, X, y=None):
        try:
            return self.pipeline.fit_transform(X, y)
        except Exception as e:
            raise RuntimeError(f"Error occurred during pipeline fit_transform: {str(e)}")

    @staticmethod
    def preprocess_target(y):
        if pd.api.types.is_object_dtype(y):
            labelencoder = LabelEncoder()
            y = pd.Series(labelencoder.fit_transform(y))
        return y

    def process(self, X, y):
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
            raise ValueError("X must be a pandas DataFrame and y must be a pandas Series")

        y.dropna(inplace=True)
        X = X.iloc[y.index].reset_index(drop=True)
        y = y.reset_index(drop=True)

        try:
            X = self.fit_transform(X)
        except Exception as e:
            raise RuntimeError(f"Error occurred while processing X: {str(e)}")

        y = y.iloc[X.index]
        y = self.preprocess_target(y)
        
        print("Data after processing:", X.shape, " and ", y.shape)
        return X, y