from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap

from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model

import missing_value as mv
import pandas as pd
import numpy as np


def standardize(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    scaled_data = pd.DataFrame(df_scaled, columns=df.columns)
    return scaled_data


def build_autoencoder(input_dim, encoding_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder, encoder


def apply_autoencoder(X, encoding_dim=10, epochs=50, batch_size=256):
    input_dim = X.shape[1]
    autoencoder, encoder = build_autoencoder(input_dim, encoding_dim)
    
    autoencoder.fit(X, X, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)
    
    X_reduced = encoder.predict(X)
    return X_reduced


def is_pca_valid(X):
    correlation_matrix = np.corrcoef(X, rowvar=False)
    determinant = np.linalg.det(correlation_matrix)
    
    if determinant < 0.01:
        return True  # The determinant of the correlation matrix suggests PCA is appropriate
    else:
        return False  # The determinant of the correlation matrix suggests PCA may not be appropriate


def num_components_for_variance(X, variance_threshold=0.95):
    pca = PCA()
    pca.fit(X)
    
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    num_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    return num_components


def apply_pca(X):
    num_components = num_components_for_variance(X)
    pca = PCA(n_components=num_components)
    
    X_reduced = pca.fit_transform(X)
    return X_reduced


def determine_umap_params(X, min_neighbors=5, max_neighbors=50):
    n_samples, n_features = X.shape
    n_components = num_components_for_variance(X)
    n_neighbors = min(max(min_neighbors, int(n_samples // 100)), max_neighbors)
    min_dist = 0.1 
    
    return n_components, n_neighbors, min_dist


def apply_umap(X):
    n_components, n_neighbors, min_dist = determine_umap_params(X)
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    X_reduced = reducer.fit_transform(X)
    return X_reduced


def callingfunc():
    try:
        X, y = mv.callingfunc()
        X_scaled_df = standardize(X)
        print(f"Shape of feature before feature selection:  {X.shape}")
        
        if is_pca_valid(X_scaled_df):
            X_reduced = apply_pca(X_scaled_df)
        else:
            if X_scaled_df.shape[0] <= 10000 and X_scaled_df.shape[1] <= 50:  # Assuming UMAP is better for smaller datasets
                X_reduced = apply_umap(X_scaled_df)
            else:
                X_reduced = apply_autoencoder(X_scaled_df)
        
        print(f"Shape of feature after reduction: {X_reduced.shape}")
    except Exception as e:
        print(f"An error occurred: {e}")


callingfunc()