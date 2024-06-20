import pandas as pd
import numpy as np
from scipy.stats import shapiro, kstest, anderson, jarque_bera, skew, kurtosis
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import DBSCAN
from math import sqrt
from diptest import diptest



"""
    parameters:
        data: data should be of pandas series object and should consist of numerical data only, if not get converted into series object.
        minlen: the parameter specially for the test name 'shapiro' which specify that data should have size atleast of 5000 (default).
        tests: this parameter is list of string with name of tests that user can perform
"""



def is_normal_distribution(data, minlen = 5000, tests = ['skew-kurtosis']):
    # Convert data to pandas Series if it's not already
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    
    numerical_type = ['int64', 'int32', 'int16', 'int8', 'uint64', 'uint32', 'uint16', 'uint8', 'float64', 'float32']
    
    if data.dtype not in numerical_type:
        raise ValueError(f"data should contain only numerical values permitted data types are {numerical_type}")
    
    if not isinstance(tests, list):
        raise ValueError("tests should should be an list of string")

    test_list = ['skew-kurtosis', 'shapiro', 'kstest', 'anderson', 'jarque-bera']

    for t in tests:
        if t not in test_list:
            raise ValueError(f"Invalid test. did you want to write {test_list}")

    # Early exit if sample size is too large for Shapiro-Wilk (recommended max is 5000)
    if (len(data) < minlen) or ('shaprio' in tests):
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


"""
    parameters:
        methodname: a string having value of name of outlier methods that are provided.
"""



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


"""
    parameters:
        data: data should be of pandas dataframe object and should consist of numerical data only, if not get converted into series object.
        tests: this parameter is list of string with name of tests that user can perform
        method: which method user want to use to handle the outliers
        handle: whether user want to trim the data or cap the data
"""



def handle_outliers(data, tests = ['skew-kurtosis'], method = 'default', handle = "capping"):
    
    """
        error handling
    """

    numerical_type = ['int64', 'int32', 'int16', 'int8', 'uint64', 'uint32', 'uint16', 'uint8', 'float64', 'float32']

    if isinstance(data, pd.DataFrame) == False:
        raise ValueError(f"Invalid datatype {type(data)} this function accept pandas dataframe")

    for i in data.columns:
        if data[i].dtype not in numerical_type:
            raise ValueError(f"Invalid dtype, this module can work with only {numerical_type} dtypes")

    
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

    method_list = ['isolation-forest', 'lof', 'default']

    if method not in method_list:
        raise ValueError(f"Invalid test. did you want to write {method_list}")


    """
        error handling finishes here
    """

    if (method == 'default' and data.shape[0] >= 10000) or (method == 'isolation-forest'):
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
        dip, dip_p_value = diptest(distances)

        use_lof = (dist_variance > variance_threshold) and (dip_p_value < dip_p_value_threshold)

    if use_lof or method == 'lof':
        lof = LocalOutlierFactor(n_neighbors=k)
        y_pred = lof.fit_predict(data)
        outliers = data[y_pred == -1]
        cleaned_data = data[y_pred != -1]
        return outliers, cleaned_data

    if method == 'default':
        outliers = pd.DataFrame()
        cleaned_data = pd.DataFrame()

        for column in data.columns:
            if is_normal_distribution(data[column]):
                mean = data[column].mean()
                std = data[column].std()

                upper_limit = mean + 3*std
                lower_limit = mean - 3*std
                
                outliers[column] = data[(data[column] > upper_limit) | (data[column] < lower_limit)][column]
                cleaned_data[column] = data[(data[column] < upper_limit) & (data[column] > lower_limit)][column]
            
            else:
                q25 = data[column].quantile(0.25)
                q75 = data[column].quantile(0.75)

                iqr = q75 - q25

                upper_limit = q75 + 1.5 * iqr
                lower_limit = q25 - 1.5 * iqr

                outliers[column] = data[(data[column] > upper_limit) | (data[column] < lower_limit)][column]
                cleaned_data[column] = data[(data[column] < upper_limit) & (data[column] > lower_limit)][column]
        return outliers, cleaned_data
    

data = pd.read_csv('data\placement.csv')
outliers, cleaned = handle_outliers(data.drop(['placed'], axis=1))
print(outliers)