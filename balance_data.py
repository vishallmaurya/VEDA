import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN, SMOTEN
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from collections import Counter
from sklearn.cluster import DBSCAN
from scipy.stats import entropy

def calculate_dynamic_threshold(y):
    """
    Calculate a dynamic threshold based on class distribution.
    """
    class_ratios = np.array(list(Counter(y).values())) / len(y)
    iqr = np.percentile(class_ratios, 75) - np.percentile(class_ratios, 25)
    dynamic_threshold = np.median(class_ratios) - (iqr / 2)
    return max(0.1, dynamic_threshold)

def calculate_entropy(y):
    """
    Calculate the entropy of the class distribution.
    """
    class_probs = np.array(list(Counter(y).values())) / len(y)
    return entropy(class_probs, base=2)


def estimate_class_density(X, y):
    """
    Estimate the density ratio of minority to majority class using DBSCAN clustering.
    """
    db = DBSCAN().fit(X)
    labels = db.labels_

    # Ensure there are points labeled as 0 and 1
    majority_density = np.sum(labels == 0) / len(y[y == 0]) if len(y[y == 0]) > 0 else 1
    minority_density = np.sum(labels == 1) / len(y[y == 1]) if len(y[y == 1]) > 0 else 1

    # Prevent division by zero by ensuring densities are not zero
    density_ratio = (minority_density / majority_density) if majority_density > 0 else 1

    return density_ratio


def adaptive_threshold(X, y):
    """
    Calculate adaptive thresholds using percentiles of class ratios.
    """
    n_samples = len(y)
    class_ratios = np.array(list(Counter(y).values())) / n_samples
    small_size = np.percentile(class_ratios, 10) * n_samples
    large_size = np.percentile(class_ratios, 90) * n_samples
    imbalance_threshold = np.percentile(class_ratios, 25)
    
    return small_size, large_size, imbalance_threshold

def select_balancing_strategy(X, y, threshold=0.5):
    """
    Select the most suitable balancing strategy based on data characteristics.
    """
    # Adaptive thresholds based on the data
    small_size, large_size, imbalance_threshold = adaptive_threshold(X, y)

    # Check initial class distribution
    counter = Counter(y)
    majority_class = max(counter, key=counter.get)
    minority_class = min(counter, key=counter.get)
    majority_count = counter[majority_class]
    minority_count = counter[minority_class]
    imbalance_ratio = minority_count / majority_count

    # Calculate additional metrics for more informed decisions
    dynamic_threshold = calculate_dynamic_threshold(y)
    entropy_value = calculate_entropy(y)
    density_ratio = estimate_class_density(X, y)

    # Adjust strategy selection using dynamic thresholds and additional metrics
    if imbalance_ratio >= max(threshold, dynamic_threshold):
        return "none", None

    if density_ratio < 0.5:
        strategy = "oversample"
        sampler = SMOTEN() if entropy_value > 0.5 else SMOTE()
    elif len(X) < small_size:
        strategy = "oversample"
        sampler = ADASYN() if imbalance_ratio > 0.2 else SMOTE()
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

def balance_data(X, y, threshold=0.5):
    """
    Balance the dataset using the most suited strategy based on its nature.
    """
    # Select the most suitable strategy
    strategy, sampler = select_balancing_strategy(X, y, threshold)
    
    if strategy == "none":
        print("Data is already balanced or within the acceptable threshold.")
        return X, y, strategy, None

    print(f"Applying {strategy} strategy for balancing.")
    
    # Apply the selected strategy
    if strategy in ["oversample", "combine"]:
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

def callingfunc(X, y, classification):
    if classification == None:
        raise ValueError("Parameter classification can't be empty")
    if classification == True:
        return balance_data(X, y)
    return X, y, None, None

# Example usage:
# X, y = pd.DataFrame(np.random.randn(10000, 10)), pd.Series(np.random.choice([0, 1], size=10000, p=[0.95, 0.05]))
# X_res, y_res, used_strategy, model = balance_data(X, y)
