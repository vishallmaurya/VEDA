import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap.umap_ as umap

from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd
from sklearn.pipeline import Pipeline


class Standardizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
    
    def fit(self, X, y=None):
        try:
            if not isinstance(X, (pd.DataFrame, np.ndarray)):
                raise TypeError("Input data X should be a pandas DataFrame or numpy array.")
            
            if X.isnull().values.any():
                raise ValueError("Input data X contains NaN values, which cannot be handled by StandardScaler.")

            self.scaler.fit(X)
        except (TypeError, ValueError) as e:
            raise RuntimeError(f"Failed to fit StandardScaler due to invalid input data: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while fitting StandardScaler: {e}")
        return self
    
    def transform(self, X):
        try:
            if not isinstance(X, (pd.DataFrame, np.ndarray)):
                raise TypeError("Input data X should be a pandas DataFrame or numpy array.")
            
            if X.isnull().values.any():
                raise ValueError("Input data X contains NaN values, which cannot be handled by StandardScaler.")
            
            X_scaled = self.scaler.transform(X)
            return pd.DataFrame(X_scaled, columns=X.columns)
        except (TypeError, ValueError) as e:
            raise RuntimeError(f"Failed to transform data with StandardScaler due to invalid input data: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while transforming data with StandardScaler: {e}")

    def fit_transform(self, X, y=None):
        try:
            self.fit(X, y)
            return self.transform(X)
        except Exception as e:
            raise RuntimeError(f"An error occurred during fit_transform: {e}")
        

class PCATransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variance_threshold=0.95, n_components = None):
        self.variance_threshold = variance_threshold
        self.pca = None

        if n_components is None:
            raise ValueError("n_components must be an integer")
        self.n_components = n_components
    
    def fit(self, X, y=None):
        try:
            if not isinstance(X, (pd.DataFrame, np.ndarray)):
                raise TypeError("Input data X should be a pandas DataFrame or numpy array.")
            
            if X.isnull().values.any():
                raise ValueError("Input data X contains NaN values, which cannot be handled by PCA.")
            
            if self.n_components == 0:
                raise ValueError("No components explain the desired variance threshold.")
                
        except (TypeError, ValueError) as e:
            raise RuntimeError(f"Failed to fit PCA due to invalid input data: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while fitting PCA: {e}")
        return self
    
    def transform(self, X):
        try:
            if not isinstance(X, (pd.DataFrame, np.ndarray)):
                raise TypeError("Input data X should be a pandas DataFrame or numpy array.")
            
            if X.isnull().values.any():
                raise ValueError("Input data X contains NaN values, which cannot be handled by PCA.")
            
            if self.n_components is None:
                raise RuntimeError("The transformer has not been fitted yet.")
            
            pca = PCA(n_components=self.n_components)
            return pca.fit_transform(X)
        except (TypeError, ValueError) as e:
            raise RuntimeError(f"Failed to transform data with PCA due to invalid input data: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while transforming data with PCA: {e}")
    
    def fit_transform(self, X, y=None):
        try:
            self.fit(X, y)
            return self.transform(X)
        except Exception as e:
            raise RuntimeError(f"An error occurred during fit_transform: {e}")
        


class UMAPTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, prioritize_reproducibility=True, min_neighbors=5, max_neighbors=50, n_components=None):
        self.prioritize_reproducibility = prioritize_reproducibility
        self.min_neighbors = min_neighbors
        self.max_neighbors = max_neighbors
        self.reducer = None

        if n_components is None:
            raise ValueError("n_components must contain some integer value.")

        self.n_components = n_components
    
    def fit(self, X, y=None):
        try:
            if not isinstance(X, (pd.DataFrame, np.ndarray)):
                raise TypeError("Input data X should be a pandas DataFrame or numpy array.")
            
            if X.isnull().values.any():
                raise ValueError("Input data X contains NaN values, which cannot be handled by UMAP.")
            
            n_samples, _ = X.shape
            n_neighbors = min(max(self.min_neighbors, int(n_samples // 100)), self.max_neighbors)
            
            self.reducer = umap.UMAP(n_components=self.num_components,
                                     n_neighbors=n_neighbors,
                                     min_dist=0.1,
                                     random_state=42 if self.prioritize_reproducibility else None,
                                     n_jobs=1 if self.prioritize_reproducibility else -1)
        except (TypeError, ValueError) as e:
            raise RuntimeError(f"Failed to initialize UMAP due to invalid input data: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while initializing UMAP: {e}")
        return self
    
    def transform(self, X):
        try:
            if not isinstance(X, (pd.DataFrame, np.ndarray)):
                raise TypeError("Input data X should be a pandas DataFrame or numpy array.")
            
            if X.isnull().values.any():
                raise ValueError("Input data X contains NaN values, which cannot be handled by UMAP.")
            
            if self.reducer is None:
                raise RuntimeError("The transformer has not been fitted yet.")
            
            return self.reducer.fit_transform(X)
        except (TypeError, ValueError) as e:
            raise RuntimeError(f"Failed to transform data with UMAP due to invalid input data: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while transforming data with UMAP: {e}")
    
    def fit_transform(self, X, y=None):
        try:
            self.fit(X, y)
            return self.transform(X)
        except Exception as e:
            raise RuntimeError(f"An error occurred during fit_transform: {e}")
        


class AutoencoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variance_threshold=0.95, min_dim=10, max_dim=100, 
                 hidden_layers=1, optimizer='adam', loss='mean_squared_error', 
                 min_epochs=20, max_epochs=100, min_batch_size=32, max_batch_size=256, encoding_dim = None):
        self.variance_threshold = variance_threshold
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.hidden_layers = hidden_layers
        self.optimizer = optimizer
        self.loss = loss
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.autoencoder = None
        self.encoder = None

        if encoding_dim is None:
            raise ValueError("encoding_dim must have some integer value")

        self.encoding_dim = encoding_dim
    
    def _determine_encoding_dim(self, X):
        try:
            self.encoding_dim = max(self.min_dim, min(self.encoding_dim, self.max_dim))
            return self.encoding_dim
        except Exception as e:
            raise RuntimeError(f"Failed to determine encoding dimension: {e}")

    def _determine_epochs(self, n_samples):
        try:
            if n_samples < 5000:
                return self.min_epochs
            elif n_samples < 10000:
                return int(self.min_epochs + (self.max_epochs - self.min_epochs) * (n_samples - 5000) / 5000)
            else:
                return self.max_epochs
        except Exception as e:
            raise RuntimeError(f"Failed to determine epochs: {e}")
    
    def _determine_batch_size(self, n_samples):
        try:
            if n_samples < 5000:
                return self.min_batch_size
            elif n_samples < 20000:
                return int(self.min_batch_size + (self.max_batch_size - self.min_batch_size) * (n_samples - 5000) / 15000)
            else:
                return self.max_batch_size
        except Exception as e:
            raise RuntimeError(f"Failed to determine batch size: {e}")
    
    def _build_autoencoder(self, input_dim, encoding_dim):
        try:
            input_layer = Input(shape=(input_dim,))
            encoded = Dense(encoding_dim, activation='relu')(input_layer)
            x = encoded
            for _ in range(self.hidden_layers - 1):
                x = Dense(encoding_dim // 2, activation='relu')(x)
            decoded = Dense(input_dim, activation='sigmoid')(x)
            
            autoencoder = Model(input_layer, decoded)
            encoder = Model(input_layer, encoded)
            
            autoencoder.compile(optimizer=self.optimizer, loss=self.loss)
            return autoencoder, encoder
        except Exception as e:
            raise RuntimeError(f"Failed to build autoencoder model: {e}")
    
    def fit(self, X, y=None):
        try:
            input_dim = X.shape[1]
            encoding_dim = self._determine_encoding_dim(X)
            self.autoencoder, self.encoder = self._build_autoencoder(input_dim, encoding_dim)
            
            epochs = self._determine_epochs(X.shape[0])
            batch_size = self._determine_batch_size(X.shape[0])
            early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
            
            self.autoencoder.fit(X, X, epochs=epochs, batch_size=batch_size, shuffle=True, 
                                 verbose=1, callbacks=[early_stopping])
        except Exception as e:
            raise RuntimeError(f"Failed to fit autoencoder: {e}")
        return self
    
    def transform(self, X):
        try:
            X_reduced = self.encoder.predict(X)
            return X_reduced
        except Exception as e:
            raise RuntimeError(f"Failed to transform data using encoder: {e}")

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)



class DimensionReducer:
    def __init__(self,  variance_threshold=0.95,
                 prioritize_reproducibility=True, min_neighbors=5, max_neighbors=50,
                 min_dim=10, max_dim=100, 
                 hidden_layers=1, optimizer='adam', loss='mean_squared_error', 
                 min_epochs=20, max_epochs=100, min_batch_size=32, max_batch_size=256):
        
        self.variance_threshold = variance_threshold
        self.prioritize_reproducibility = prioritize_reproducibility
        self.min_neighbors = min_neighbors
        self.max_neighbors = max_neighbors
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.hidden_layers = hidden_layers
        self.optimizer = optimizer
        self.loss = loss
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.n_components = None
        self.skip = None

    def _is_pca_valid(self, X):
        try:
            correlation_matrix = np.corrcoef(X, rowvar=False)
            determinant = np.linalg.det(correlation_matrix)
            
            if determinant < 0.01:
                return True  # The determinant of the correlation matrix suggests PCA is appropriate
            else:
                return False  # The determinant of the correlation matrix suggests PCA may not be appropriate
        except Exception as e:
            raise RuntimeError(f"Failed to validate PCA: {e}")

    def fit(self, X, y=None):
        try:
            if not isinstance(X, pd.DataFrame):
                raise ValueError(f"Expected a pandas DataFrame, but got {type(X).__name__}.")
        except ValueError as e:
            raise RuntimeError(f"Input validation error: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred during fit: {e}")
        return self
    
    def transform(self, X, y):
        try:
            if not isinstance(X, pd.DataFrame):
                raise ValueError(f"Expected a pandas DataFrame, but got {type(X).__name__}.")

            pca = PCA()
            pca.fit(X)
            
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            self.num_components = np.argmax(cumulative_variance >= self.variance_threshold) + 1

            if self.num_components < 30:
                self.skip = True  # Indicate that reduction is not needed and return the number of components
            else:
                self.skip = False
            
            standardizer = Standardizer()
            X_scaled_df = standardizer.fit_transform(X, y)

            if(self.skip or X.shape[1] < 30):
                return X_scaled_df, y
            
            if self._is_pca_valid(X_scaled_df):
                pca_transformer = PCATransformer(n_components=self.num_components, variance_threshold=self.variance_threshold)
                X_pca = pca_transformer.fit_transform(X_scaled_df, y)
                return X_pca, y
            else:
                if X_scaled_df.shape[0] <= 10000 and X_scaled_df.shape[1] <= 50:
                    umap_transformer = UMAPTransformer(prioritize_reproducibility=self.prioritize_reproducibility,
                                               min_neighbors=self.min_neighbors,
                                               max_neighbors=self.max_neighbors,
                                               n_components=self.num_components)
                    X_umap = umap_transformer.fit_transform(X_scaled_df, y)
                    return X_umap, y
                else:
                    auto_transformer = AutoencoderTransformer(variance_threshold=self.variance_threshold,
                                                            min_dim=self.min_dim, max_dim=self.max_dim, 
                                                            hidden_layers=self.hidden_layers, optimizer=self.optimizer,
                                                            loss=self.loss, min_epochs=self.min_epochs, max_epochs=self.max_epochs,
                                                            min_batch_size=self.min_batch_size, max_batch_size=self.max_batch_size,
                                                            encoding_dim = self.encoding_dim)
                    X_reduced = auto_transformer.fit_transform(X_scaled_df, y)
                    return X_reduced, y
        except (ValueError, TypeError) as e:
            raise RuntimeError(f"Data validation or type error in transform: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred during transform: {e}")


    def fit_transform(self, X, y=None):
        try:
            self.fit(X, y)
            return self.transform(X, y)
        except Exception as e:
            raise RuntimeError(f"An error occurred during fit_transform: {e}")