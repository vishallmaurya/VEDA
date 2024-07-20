from sklearn.preprocessing import StandardScaler
import missing_value as mv
from sklearn.decomposition import PCA
from factor_analyzer import calculate_kmo
import pandas as pd
import numpy as np

def standardize(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    scaled_data = pd.DataFrame(df_scaled, columns=df.columns)
    return scaled_data


def is_pca_valid(X):
    correlation_matrix = np.corrcoef(X, rowvar=False)
    determinant = np.linalg.det(correlation_matrix)
    
    print(f"Determinant of the correlation matrix: {determinant}")
    if determinant < 0.01:  # A small determinant suggests multicollinearity
        print("The determinant of the correlation matrix suggests PCA is appropriate.")
        return True
    else:
        print("The determinant of the correlation matrix suggests PCA may not be appropriate.")
        return False


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

def callingfunc():
    X, y = mv.callingfunc()
    print(f"Shape of feature before feature selection:  {X.shape}")

    X_scaled_df = standardize(X)
    if(is_pca_valid(X_scaled_df)):
        X_reduced = apply_pca(X_scaled_df)
    else:
        pass
    print(f"Shape of feature before feature selection:  {X_reduced.shape}")

callingfunc()