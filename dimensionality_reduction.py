import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap.umap_ as umap

from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import EarlyStopping

import pandas as pd


def standardize(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    scaled_data = pd.DataFrame(df_scaled, columns=df.columns)
    return scaled_data


def determine_encoding_dim(X, variance_threshold=0.95, min_dim=10, max_dim=100):
    pca = PCA(n_components=X.shape[1])
    pca.fit(X)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    encoding_dim = np.argmax(cumulative_variance >= variance_threshold) + 1
    encoding_dim = max(min_dim, min(encoding_dim, max_dim))
    return encoding_dim


def determine_epochs(n_samples, min_epochs=20, max_epochs=100):
    if n_samples < 5000:
        return min_epochs
    elif n_samples < 10000:
        return int(min_epochs + (max_epochs - min_epochs) * (n_samples - 5000) / 5000)
    else:
        return max_epochs
    
def determine_batch_size(n_samples, min_batch_size=32, max_batch_size=256):
    if n_samples < 5000:
        return min_batch_size
    elif n_samples < 20000:
        return int(min_batch_size + (max_batch_size - min_batch_size) * (n_samples - 5000) / 15000)
    else:
        return max_batch_size


def build_autoencoder(input_dim, encoding_dim, hidden_layers=1, optimizer='adam', loss='mean_squared_error'):
    input_layer = Input(shape=(input_dim,))
    
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    x = encoded
    for _ in range(hidden_layers - 1):
        x = Dense(encoding_dim // 2, activation='relu')(x)
    
    decoded = Dense(input_dim, activation='sigmoid')(x)
    
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    
    autoencoder.compile(optimizer=optimizer, loss=loss)
    return autoencoder, encoder



def apply_autoencoder(X):
    input_dim = X.shape[1]
    encoding_dim = determine_encoding_dim(X)
    autoencoder, encoder = build_autoencoder(input_dim, encoding_dim)
    
    epochs = determine_epochs(X.shape[0])
    batch_size = determine_batch_size(X.shape[0])
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    
    autoencoder.fit(X, X, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1, callbacks=[early_stopping])
    
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


def apply_umap(X, prioritize_reproducibility=True):
    n_components, n_neighbors, min_dist = determine_umap_params(X)
    
    if prioritize_reproducibility:
        reducer = umap.UMAP(n_components=n_components, 
                            n_neighbors=n_neighbors, 
                            min_dist=min_dist, 
                            random_state=42, 
                            n_jobs=1)  # Ensures reproducibility
    else:
        reducer = umap.UMAP(n_components=n_components, 
                            n_neighbors=n_neighbors, 
                            min_dist=min_dist, 
                            random_state=None,  # No seed for parallelism
                            n_jobs=-1)  # Use all available cores for speed

    X_reduced = reducer.fit_transform(X)
    return X_reduced


def callingfunc(X, y):
    try:
        X_scaled_df = standardize(X)
        print(f"Shape of feature before feature selection:  {X.shape}")
        
        if is_pca_valid(X_scaled_df):
            X_reduced = apply_pca(X_scaled_df)
            return X_reduced, y
        else:
            if X_scaled_df.shape[0] <= 10000 and X_scaled_df.shape[1] <= 50:
                X_reduced = apply_umap(X_scaled_df)
            else:
                X_reduced = apply_autoencoder(X_scaled_df)
            return X_reduced, y
        print(f"Shape of feature after reduction: {X_reduced.shape}")
    except Exception as e:
        print(f"An error occurred: {e}")
