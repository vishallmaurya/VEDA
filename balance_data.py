import numpy as np
import pandas as pd
import imblearn
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from collections import Counter

def select_balancing_strategy(X, y, threshold=0.1, small_size=10000, large_size=100000, imbalance_threshold=0.05):
    """
    Select the most suitable balancing strategy based on data characteristics.
    
    Parameters:
    - X: Feature matrix (pd.DataFrame or np.ndarray).
    - y: Target vector (pd.Series or np.ndarray).
    - threshold: The maximum tolerated imbalance ratio (minority/majority).
    - small_size: Maximum size of a dataset considered small.
    - large_size: Minimum size of a dataset considered large.
    - imbalance_threshold: Threshold to determine if data is highly imbalanced.
    
    Returns:
    - strategy: Selected strategy ("oversample", "undersample", "combine", "anomaly", "ensemble", "none").
    - sampler: The resampler object or model if applicable, else None.
    """
    
    # Check initial class distribution
    counter = Counter(y)
    majority_class = max(counter, key=counter.get)
    minority_class = min(counter, key=counter.get)
    majority_count = counter[majority_class]
    minority_count = counter[minority_class]
    imbalance_ratio = minority_count / majority_count

    # If data is already balanced or within the acceptable threshold
    if imbalance_ratio >= threshold:
        return "none", None
    print("hiii from balance_data.py")
    # Strategy selection based on data size and imbalance ratio
    if len(X) < small_size:
        strategy = "oversample"
        sampler = SMOTE() if imbalance_ratio > 0.2 else ADASYN()
    elif len(X) > large_size:
        strategy = "combine"
        sampler = SMOTEENN() if imbalance_ratio > imbalance_threshold else SMOTETomek()
    elif imbalance_ratio < imbalance_threshold:
        strategy = "anomaly"
        sampler = IsolationForest(contamination=imbalance_ratio)
    else:
        strategy = "ensemble"
        sampler = RandomForestClassifier(class_weight='balanced_subsample')
    
    return strategy, sampler

def balance_data(X, y, threshold=0.1, small_size=10000, large_size=100000, imbalance_threshold=0.05):
    """
    Balance the dataset using the most suited strategy based on its nature.
    
    Parameters:
    - X: Feature matrix (pd.DataFrame or np.ndarray).
    - y: Target vector (pd.Series or np.ndarray).
    - threshold: The maximum tolerated imbalance ratio (minority/majority).
    - small_size: Maximum size of a dataset considered small.
    - large_size: Minimum size of a dataset considered large.
    - imbalance_threshold: Threshold to determine if data is highly imbalanced.
    
    Returns:
    - X_res: Resampled feature matrix.
    - y_res: Resampled target vector.
    - strategy: The strategy used for balancing.
    - model: Trained model if anomaly detection or ensemble method is used, otherwise None.
    """
    
    # Select the most suitable strategy
    strategy, sampler = select_balancing_strategy(X, y, threshold, small_size, large_size, imbalance_threshold)
    
    if strategy == "none":
        print("Data is already balanced or within the acceptable threshold.")
        return X, y, strategy, None

    print(f"Applying {strategy} strategy for balancing.")
    
    # Apply the selected strategy
    if strategy == "oversample":
        X_res, y_res = sampler.fit_resample(X, y)
        return X_res, y_res, strategy, None
    elif strategy == "combine":
        X_res, y_res = sampler.fit_resample(X, y)
        return X_res, y_res, strategy, None
    elif strategy == "anomaly":
        model = sampler.fit(X) 
        y_res = model.predict(X)  
        return X, y_res, strategy, model
    elif strategy == "ensemble":
        model = sampler
        model.fit(X, y)
        return X, y, strategy, model
    
def callingfunc(X, y):
    return balance_data(X, y)

# # Example usage
# X, y = pd.DataFrame(np.random.randn(10000, 10)), pd.Series(np.random.choice([0, 1], size=10000, p=[0.95, 0.05]))
# X_res, y_res, used_strategy, model = balance_data(X, y)
