import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2, mutual_info_classif, SelectKBest
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from concurrent.futures import ThreadPoolExecutor
import missing_value as mv

def standardize(df):
    """Standardizes the dataframe by scaling features to zero mean and unit variance."""
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return pd.DataFrame(df_scaled, columns=df.columns)

def select_correlation_features(X, y, percentile=90):
    """Selects features based on correlation with the target variable."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    
    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series.")
    
    if len(X) != len(y):
        raise ValueError("The number of samples in X and y must be the same.")
    
    if not (0 <= percentile <= 100):
        raise ValueError("Percentile must be between 0 and 100.")
    
    correlations = X.corrwith(y).abs()
    threshold = np.percentile(correlations, percentile)
    selected_features = correlations[correlations > threshold].index.tolist()
    return selected_features


def select_optimal_k_mi(X, y):
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

    max_k = X.shape[1]
    mi_scores = mutual_info_classif(X, y)
    
    if len(mi_scores) != X.shape[1]:
        raise RuntimeError("Mismatch between number of features and mutual information scores computed.")

    sorted_indices = np.argsort(mi_scores)[::-1]
    sorted_mi_scores = mi_scores[sorted_indices]
    
    cumulative_mi_scores = np.cumsum(sorted_mi_scores)
    normalized_cumulative_mi_scores = cumulative_mi_scores / cumulative_mi_scores[-1]
    optimal_k = np.argmax(normalized_cumulative_mi_scores >= 0.9) + 1

    optimal_k = min(optimal_k, max_k)
    return optimal_k

def select_mi_features(X, y):
    """Selects features based on mutual information."""
    k = select_optimal_k_mi(X, y)

    mi_selector = SelectKBest(mutual_info_classif, k=k)
    mi_selector.fit(X, y)
    selected_features = mi_selector.get_support(indices=True)
    return selected_features

def select_lasso_features(X, y):
    """Selects features based on Lasso regression."""
    
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

    try:
        lasso_cv = LassoCV(cv=5)
        lasso_cv.fit(X, y)
        
        best_alpha = lasso_cv.alpha_
        
        lasso = Lasso(alpha=best_alpha)
        lasso.fit(X, y)
        
        selected_features = [i for i, coef in enumerate(lasso.coef_) if coef != 0]
        
        return selected_features
    
    except ValueError as ve:
        raise ValueError(f"Value error occurred: {ve}")
    except TypeError as te:
        raise TypeError(f"Type error occurred: {te}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")


def select_aic_bic_features(X, y):
    """Selects features based on AIC and BIC criteria."""
    
    if not isinstance(X, (pd.DataFrame, np.ndarray)):
        raise TypeError("X must be a pandas DataFrame or a numpy ndarray.")
    
    if not isinstance(y, (pd.Series, np.ndarray)):
        raise TypeError("y must be a pandas Series or a numpy ndarray.")
    
    if len(X) != len(y):
        raise ValueError("The number of samples in X and y must be the same.")
    
    if isinstance(X, pd.DataFrame):
        initial_features = X.columns.tolist()
    else:
        initial_features = [f'feature_{i}' for i in range(X.shape[1])]
    
    X = np.array(X) if isinstance(X, pd.DataFrame) else X
    
    def compute_aic_bic(X, y):
        try:
            if isinstance(y, np.ndarray):
                y = pd.Series(y)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            X_train_sm = sm.add_constant(X_train)
            X_test_sm = sm.add_constant(X_test)
            
            model = sm.OLS(y_train, X_train_sm).fit()
            
            predictions = model.predict(X_test_sm)
            aic = model.aic
            bic = model.bic
            r2 = r2_score(y_test, predictions)
            
            return aic, bic, model, r2
        
        except Exception as e:
            raise RuntimeError(f"An error occurred during model fitting or evaluation: {e}")
    
    best_aic_features = initial_features[:]
    best_bic_features = initial_features[:]
    
    try:
        best_aic, _, _, best_aic_score = compute_aic_bic(X[best_aic_features], y)
        _, best_bic, _, best_bic_score = compute_aic_bic(X[best_bic_features], y)
    except Exception as e:
        raise RuntimeError(f"An error occurred while computing initial AIC/BIC: {e}")
    
    while True:
        improved = False
        for feature in initial_features:
            temp_aic_features = best_aic_features[:]
            temp_bic_features = best_bic_features[:]
            
            if feature in temp_aic_features:
                temp_aic_features.remove(feature)
            if feature in temp_bic_features:
                temp_bic_features.remove(feature)
                
            try:
                current_aic, _, _, _ = compute_aic_bic(X[temp_aic_features], y)
                _, current_bic, _, _ = compute_aic_bic(X[temp_bic_features], y)
            except Exception as e:
                raise RuntimeError(f"An error occurred while computing AIC/BIC for feature subsets: {e}")
            
            if current_aic < best_aic:
                best_aic, best_aic_features, _, _ = current_aic, temp_aic_features, _, _
                improved = True
                
            if current_bic < best_bic:
                best_bic, best_bic_features, _, _ = current_bic, temp_bic_features, _, _
                improved = True
        
        if not improved:
            break
            
    return best_aic_features, best_bic_features


def callingfunc(X, y):
    """Calls the feature selection functions and prints the selected features."""
    try:
        # y = y.values.ravel()

        X_scaled_df = standardize(X)
        print(f"Shape of feature before feature selection:  {X.shape}")

        # Perform feature selection in parallel
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(select_correlation_features, X, y): "correlation_features",
                executor.submit(select_mi_features, X_scaled_df, y): "mi_features",
                executor.submit(select_lasso_features, X_scaled_df, y): "lasso_features",
                executor.submit(select_aic_bic_features, X, y): "aic_bic_features"
            }
            results = {name: future.result() for future, name in futures.items()}

        correlation_features = results["correlation_features"]
        mi_features = results["mi_features"]
        lasso_features = results["lasso_features"]
        aic_features, bic_features = results["aic_bic_features"]

        all_selected_features = list(set(correlation_features + 
                                         [X.columns[i] for i in mi_features] + 
                                         [X.columns[i] for i in lasso_features] + 
                                         aic_features + 
                                         bic_features))

        X = X[all_selected_features]
        print(f"Selected features: {all_selected_features}")
        print(f"Shape of feature after feature selection:  {len(all_selected_features)}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return X, y