import pandas as pd
import numpy as np
from scipy.stats import shapiro, kstest, anderson, jarque_bera, skew, kurtosis
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import DBSCAN
from math import sqrt
from diptest import diptest

def is_normal_distribution(data, minlen = 5000, tests = ['skew-kurtosis']):
    # Convert data to pandas Series if it's not already
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    
    if not isinstance(tests, list):
        raise ValueError("tests should should be an list of string")

    test_list = ['skew-kurtosis', 'shapiro', 'kstest', 'anderson', 'jarque-bera']

    for t in tests:
        if t not in test_list:
            raise ValueError(f"Invalid test. did you want to write {test_list}")

    # Early exit if sample size is too large for Shapiro-Wilk (recommended max is 5000)
    if (len(data) < minlen) and ('shaprio' in tests):
        # Shapiro-Wilk Test
        shapiro_stat, shapiro_p_value = shapiro(data)
        if shapiro_p_value <= 0.05:
            return False

    # Skewness and Kurtosis Check

    if('skew-kurtosis' in tests):
        data_skewness = skew(data)
        data_kurtosis = kurtosis(data, fisher=False)
        if abs(data_skewness) >= 0.5 or abs(data_kurtosis - 3) >= 0.5:
            return False

    # Kolmogorov-Smirnov Test

    if('kstest' in tests):
        ks_stat, ks_p_value = kstest(data, 'norm', args=(data.mean(), data.std()))
        if ks_p_value <= 0.05:
            return False
    
    # Anderson-Darling Test

    if('anderson' in tests):
        anderson_result = anderson(data)
        if any(anderson_result.statistic >= cv for cv in anderson_result.critical_values):
            return False
    
    # Jarque-Bera Test

    if('jarque-bera' in tests):
        jb_stat, jb_p_value = jarque_bera(data)
        if jb_p_value <= 0.05:
            return False
    
    return True

def get_outliermethod_params(methodname):
    if not isinstance(methodname, str):
        raise ValueError("The methodname should be a string")

    method_list = ['isolation-forest', 'lof', 'dbscan', 'z-score', 'iqr']

    if methodname not in method_list:
        raise ValueError(f"There is no method defined in the library of name {methodname}")
    
    if methodname == 'isolation-forest':
        iso_forest = IsolationForest()
        return iso_forest.get_params()
    elif methodname == 'lof':
        lof = LocalOutlierFactor()
        return lof.get_params()
    elif methodname == 'dbscan':
        dbscan = DBSCAN()
        return dbscan.get_params()
    elif methodname == 'z-score':
        return None
    elif methodname == 'iqr':
        return None


def handle_outliers(data, get_outliers = False, tests = ['skew-kurtosis'], method = 'default', handle = "capping", lower_percentile = 0.03, upper_percentile = 0.97):
    
    """
        error handling
    """
    
    if not isinstance(data, pd.DataFrame):
        data = data.to_frame()
    
    if not isinstance(tests, list):
        raise ValueError("tests should should be an list of string")
    
    if not isinstance(method, str):
        raise ValueError("The method should be a string")

    if not isinstance(handle, str):
        raise ValueError("The handle should be a string")

    handle_list = ['capping', 'trimming', 'winsorization']

    if handle not in handle_list:
        raise ValueError("Possible value for the parameter handle is 'capping' ,'timming' or 'winsorization'")
    
    test_list = ['skew-kurtosis', 'shapiro', 'kstest', 'anderson', 'jarque-bera']

    for t in tests:
        if t not in test_list:
            raise ValueError(f"Invalid test. did you want to write {test_list}")

    method_list = ['isolation-forest', 'lof', 'dbscan', 'z-score', 'iqr']

    if method not in method_list:
        raise ValueError(f"Invalid test. did you want to write {method_list}")


    """
        error handling finishes here
    """

    if (method='default' and len(data) >= 10000) or (method == 'isolation-forest') :
        iso_forest = IsolationForest(contamination=0.03, n_estimators=200, max_samples=sqrt(len(data)))
        outlier = iso_forest.fit_predict(data)
        outliers = data[outlier == -1]
        cleaned_data = data[outlier != -1]
        return outliers, cleaned_data

    if method == 'default':
        k = 20
        variance_threshold = 0.05
        dip_p_value_threshold = 0.05


        nbrs = NearestNeighbors(n_neighbors=k).fit(data)
        distances, _ = nbrs.kneighbors(data)
        distances = distances[:, k-1]

        dist_variance = np.var(distances)
        dist_std = np.std(distances)

        dip, dip_p_value = diptest(distances)

        use_lof = (dist_variance > variance_threshold) and (dip_p_value < dip_p_value_threshold)

    if use_lof or method == 'lof':
        lof = LocalOutlierFactor(n_neighbors=k)
        y_pred = lof.fit_predict(X)
        outliers = data[y_pred == -1]
        cleaned_data = data[y_pred != -1]
        return outliers, cleaned_data

    #     if get_outliers == True:
    #         outliers = data[outlier == -1]
    #         return outliers, data_cleaned
    #     else:
    #         return data_cleaned
    
    # if method == 'dbscan':
    #     dbscan = DBSCAN()
    #     outlier = dbscan.fit_predict(data)
    #     data_cleaned = data[outlier != -1]

    #     if get_outliers == True:
    #         outliers = data[outlier == -1]
    #         return outliers, data_cleaned
    #     else:
    #         return data_cleaned
    
    # if method == 'z-score':
    #     mean = data.mean()
    #     std = data.std()

    #     upper_limit = mean + 3*std
    #     lower_limit = mean - 3*std

    #     outliers = data[(data > upper_limit) | (data < lower_limit)]
        
    #     if get_outliers == True:
    #         return outliers, data_cleaned
    #     else:
    #         return data_cleaned
    
    # if method == 'iqr':
    #     q25 = data.quantile(0.25)
    #     q75 = data.quantile(0.75)

    #     iqr = q75 - q25

    #     upper_limit = q75 + 1.5 * iqr
    #     lower_limit = q25 - 1.5 * iqr
        
    #     outliers = data[(data > upper_limit) | (data < lower_limit)]

    #     if get_outliers == True:
    #         return outliers, data_cleaned
    #     else:
    #         return data_cleaned


data = pd.read_csv('data\placement.csv')
print(data.shape)
cleaned = handle_outliers(data['cgpa'], method = 'isolation-forest')
print(cleaned.shape)