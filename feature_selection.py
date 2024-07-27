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
    correlations = X.corrwith(y).abs()
    threshold = np.percentile(correlations, percentile)
    selected_features = correlations[correlations > threshold].index.tolist()
    return selected_features

def select_optimal_k_chi2(X, y):
    """Selects the optimal number of features based on the chi-squared test."""
    max_k = X.shape[1]

    # Ensure all values in X are non-negative for chi2 test
    X_non_negative = X - X.min().min()
    chi2_scores, _ = chi2(X_non_negative, y)
    
    sorted_indices = np.argsort(chi2_scores)[::-1]
    sorted_chi2_scores = chi2_scores[sorted_indices]
    
    cumulative_chi2_scores = np.cumsum(sorted_chi2_scores)
    normalized_cumulative_chi2_scores = cumulative_chi2_scores / cumulative_chi2_scores[-1]
    
    optimal_k = np.argmax(normalized_cumulative_chi2_scores >= 0.9) + 1
    optimal_k = min(optimal_k, max_k)

    return optimal_k

def select_chi2_features(X, y):
    """Selects features based on the chi-squared test."""
    k = select_optimal_k_chi2(X, y)
    
    # Ensure all values in X are non-negative for chi2 test
    X_non_negative = X - X.min().min()
    chi2_selector = SelectKBest(chi2, k=k)
    chi2_selector.fit(X_non_negative, y)
    selected_features = chi2_selector.get_support(indices=True)
    return selected_features

def select_optimal_k_mi(X, y):
    """Selects the optimal number of features based on mutual information."""
    max_k = X.shape[1]
    mi_scores = mutual_info_classif(X, y)
    
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
    lasso_cv = LassoCV(cv=5)
    lasso_cv.fit(X, y)
    
    best_alpha = lasso_cv.alpha_
    
    lasso = Lasso(alpha=best_alpha)
    lasso.fit(X, y)
    
    selected_features = [i for i, coef in enumerate(lasso.coef_) if coef != 0]
    return selected_features

def select_aic_bic_features(X, y):
    """Selects features based on AIC and BIC criteria."""
    def compute_aic_bic(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train_sm = sm.add_constant(X_train)
        X_test_sm = sm.add_constant(X_test)
        
        model = sm.OLS(y_train, X_train_sm).fit()
        predictions = model.predict(X_test_sm)
        aic = model.aic
        bic = model.bic
        return aic, bic, model, r2_score(y_test, predictions)
    
    initial_features = X.columns.tolist()
    best_aic_features = initial_features[:]
    best_bic_features = initial_features[:]
    
    best_aic, best_bic, best_aic_model, best_aic_score = compute_aic_bic(X[best_aic_features], y)
    _, _, best_bic_model, best_bic_score = compute_aic_bic(X[best_bic_features], y)
    
    while True:
        improved = False
        for feature in initial_features:
            temp_aic_features = best_aic_features[:]
            temp_bic_features = best_bic_features[:]
            if feature in temp_aic_features:
                temp_aic_features.remove(feature)
            if feature in temp_bic_features:
                temp_bic_features.remove(feature)
                
            current_aic, _, current_aic_model, current_aic_score = compute_aic_bic(X[temp_aic_features], y)
            _, current_bic, current_bic_model, current_bic_score = compute_aic_bic(X[temp_bic_features], y)
            
            if current_aic < best_aic:
                best_aic, best_aic_features, best_aic_model, best_aic_score = current_aic, temp_aic_features, current_aic_model, current_aic_score
                improved = True
                
            if current_bic < best_bic:
                best_bic, best_bic_features, best_bic_model, best_bic_score = current_bic, temp_bic_features, current_bic_model, current_bic_score
                improved = True
        
        if not improved:
            break
            
    return best_aic_features, best_bic_features

def callingfunc():
    """Calls the feature selection functions and prints the selected features."""
    try:
        X, y = mv.callingfunc()
        X_scaled_df = standardize(X)
        print(f"Shape of feature before feature selection:  {X.shape}")

        # Perform feature selection in parallel
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(select_correlation_features, X, y): "correlation_features",
                executor.submit(select_chi2_features, X_scaled_df, y): "chi2_features",
                executor.submit(select_mi_features, X_scaled_df, y): "mi_features",
                executor.submit(select_lasso_features, X_scaled_df, y): "lasso_features",
                executor.submit(select_aic_bic_features, X, y): "aic_bic_features"
            }
            results = {name: future.result() for future, name in futures.items()}

        correlation_features = results["correlation_features"]
        chi2_features = results["chi2_features"]
        mi_features = results["mi_features"]
        lasso_features = results["lasso_features"]
        aic_features, bic_features = results["aic_bic_features"]

        all_selected_features = list(set(correlation_features + 
                                         [X.columns[i] for i in chi2_features] + 
                                         [X.columns[i] for i in mi_features] + 
                                         [X.columns[i] for i in lasso_features] + 
                                         aic_features + 
                                         bic_features))

        print(f"Selected features: {all_selected_features}")
        print(f"Shape of feature after feature selection:  {len(all_selected_features)}")
    except Exception as e:
        print(f"An error occurred: {e}")

callingfunc()
