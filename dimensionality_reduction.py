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
    try:
        chi_square_value, p_value = bartlett(*X.T)
        if p_value < 0.05:
            print("Bartlett's test suggests PCA is appropriate.")
            return True
        else:
            print("Bartlett's test suggests PCA may not be appropriate.")
            return False
    except Exception as e:
        print(f"Error in Bartlett's test: {e}")
        return False
    

def diagnose_data(X):
    # Check for identical rows
    unique_rows = np.unique(X, axis=0)
    if unique_rows.shape[0] != X.shape[0]:
        print("Warning: There are identical rows in the dataset.")
    
    # Check for multicollinearity
    correlation_matrix = np.corrcoef(X, rowvar=False)
    high_corr = np.abs(correlation_matrix) > 0.9
    np.fill_diagonal(high_corr, False)  # Ignore self-correlations
    if np.any(high_corr):
        print("Warning: There are highly correlated features in the dataset.")
    
    # Print summary statistics
    print(f"Data shape: {X.shape}")
    print(f"Variance of features: {np.var(X, axis=0)}")


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
    
    print(f"Shape of feature before feature selection:  {X.shape}")
    diagnose_data(X)

    diagnose_data(X_scaled_df)
    is_pca_valid(X_scaled_df)
    # if(is_pca_valid(X_scaled_df)):
        # X_reduced = apply_pca(X_scaled_df)
    # print(f"Shape of feature before feature selection:  {X_reduced.shape}")

callingfunc()