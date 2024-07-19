from sklearn.preprocessing import StandardScaler
import missing_value as mv
from sklearn.decomposition import PCA
from scipy.stats import bartlett
import pandas as pd
import numpy as np

def standardize(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    scaled_data = pd.DataFrame(df_scaled, columns=df.columns)
    return scaled_data


def is_pca_valid(X):
    chi_square_value, p_value = bartlett(*X.T)
    if p_value < 0.05:
        print("Bartlett's test suggests PCA is appropriate.")
        return True
    else:
        print("Bartlett's test suggests PCA may not be appropriate.")
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
    X_scaled_df = standardize(X)
    
    print(f"Shape of feature before feature selection:  {X.shape}")
    if(is_pca_valid(X_scaled_df)):
        X_reduced = apply_pca(X_scaled_df)
    # print(f"Shape of feature before feature selection:  {X_reduced.shape}")

callingfunc()