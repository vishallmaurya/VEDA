import pandas as pd
import numpy as np
from scipy.stats import shapiro, kstest, anderson, jarque_bera, skew, kurtosis
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import DBSCAN
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import optuna
from diptest import diptest



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



def objective(trial, X):
    contamination = trial.suggest_float('contamination', 0.01, 0.1)
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_samples = trial.suggest_int('max_samples', 1, len(X))
    random_state = 42

    # Split the data for evaluation
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=random_state)

    # Initialize and train the IsolationForest
    iso_forest = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=random_state
    )
    iso_forest.fit(X_train)
    
    # Predict on validation set
    y_pred = iso_forest.predict(X_val)
    y_true = np.ones_like(y_pred)
    y_true[y_pred == -1] = -1  # Assume that detected outliers are the minority class
    
    return f1_score(y_true, y_pred)


def calculate_optimal_trials(dataset_size):
    if dataset_size < 1000:
        return 100
    elif dataset_size < 10000:
        return 65
    elif dataset_size < 100000:
        return 40
    else:
        return 20


def tune_isolation_forest_params_with_optuna(X):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X), n_trials=calculate_optimal_trials(len(X)))

    best_params = study.best_params
    best_estimator = IsolationForest(**best_params, random_state=42)
    return best_estimator



"""
    parameters:
        data: data should be of pandas series object and should consist of numerical data only, if not get converted into series object.
        minlen: the parameter specially for the test name 'shapiro' which specify that data should have size atleast of 5000 (default).
        tests: this parameter is list of string with name of tests that user can perform
"""

def is_normal_distribution(data, minlen=5000, tests=['skew-kurtosis']):
    # Convert data to pandas Series if it's not already
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    
    numerical_type = ['int64', 'int32', 'int16', 'int8', 'uint64', 'uint32', 'uint16', 'uint8', 'float64', 'float32']
    
    if data.dtype not in numerical_type:
        raise ValueError(f"data should contain only numerical values; permitted data types are {numerical_type}")
    
    if not isinstance(tests, list):
        raise ValueError("tests should be a list of strings")

    test_list = ['skew-kurtosis', 'shapiro', 'kstest', 'anderson', 'jarque-bera']

    for t in tests:
        if t not in test_list:
            raise ValueError(f"Invalid test. Did you mean one of {test_list}?")

    if (len(data) < minlen) and ('shapiro' in tests):
        shapiro_stat, shapiro_p_value = shapiro(data)
        if shapiro_p_value <= 0.05:
            return False

    if 'skew-kurtosis' in tests:
        data_skewness = skew(data)
        data_kurtosis = kurtosis(data, fisher=False)
        if abs(data_skewness) >= 1 or abs(data_kurtosis - 3) >= 1:
            return False

    if 'kstest' in tests:
        ks_stat, ks_p_value = kstest(data, 'norm', args=(data.mean(), data.std()))
        if ks_p_value <= 0.05:
            return False
    
    if 'anderson' in tests:
        anderson_result = anderson(data)
        if any(anderson_result.statistic >= cv for cv in anderson_result.critical_values):
            return False
    
    if 'jarque-bera' in tests:
        jb_stat, jb_p_value = jarque_bera(data)
        if jb_p_value <= 0.05:
            return False
    
    return True


"""
    parameters:
        data: data should be of pandas dataframe object and should consist of numerical data only, if not get converted into series object.
        tests: this parameter is list of string with name of tests that user can perform
        method: which method user want to use to handle the outliers
        handle: whether user want to trim the data or cap the data
"""



def handle_outliers(data, y, tests=['skew-kurtosis'], method='default', handle='capping'):
    numerical_type = ['int64', 'int32', 'int16', 'int8', 'uint64', 'uint32', 'uint16', 'uint8', 'float64', 'float32']

    if not isinstance(data, pd.DataFrame):
        raise ValueError(f"Invalid datatype {type(data)}. This function accepts pandas DataFrame.")
    
    for col in data.columns:
        if data[col].dtype not in numerical_type:
            raise ValueError(f"Invalid dtype in column '{col}', expected one of {numerical_type}.")

    if not isinstance(tests, list):
        raise ValueError("tests should be a list of strings.")
    
    if not isinstance(method, str):
        raise ValueError("The method should be a string.")

    if not isinstance(handle, str):
        raise ValueError("The handle should be a string.")

    handle_list = ['capping', 'trimming', 'winsorization']
    if handle not in handle_list:
        raise ValueError("Possible values for handle are 'capping', 'trimming', or 'winsorization'.")

    test_list = ['skew-kurtosis', 'shapiro', 'kstest', 'anderson', 'jarque-bera']
    for t in tests:
        if t not in test_list:
            raise ValueError(f"Invalid test. Did you mean one of {test_list}?")

    method_list = ['isolation-forest', 'lof', 'default']
    if method not in method_list:
        raise ValueError(f"Invalid method. Did you mean one of {method_list}?")

    # Prepare to track outliers
    outlier_indices = []

    # Check for multimodal columns using Dip Test
    multimodal_columns = []
    for column in data.columns:
        dip_stat, dip_p_value = diptest(data[column])
        if dip_p_value < 0.05:
            multimodal_columns.append(column)

    if multimodal_columns:
        print(f"Multimodal distribution detected in columns: {multimodal_columns}")
        
        # Apply clustering-based approach for multimodal columns
        for column in multimodal_columns:
            dbscan = DBSCAN()
            labels = dbscan.fit_predict(data[[column]])
            noise = data[labels == -1]
            outlier_indices.extend(noise.index)

    if (method == 'default' and data.shape[0] >= 10000) or (method == 'isolation-forest'):
        print("Isolation Forest is used for outlier detection.")
        iso_forest = tune_isolation_forest_params_with_optuna(data)
        outlier = iso_forest.fit_predict(data)
        outlier_indices.extend(data.index[outlier == -1])

    if method == 'default':
        k = int(sqrt(len(data)))
        variance_threshold = 0.05
        dip_p_value_threshold = 0.05

        nbrs = NearestNeighbors(n_neighbors=k).fit(data)
        distances, _ = nbrs.kneighbors(data)
        distances = distances[:, k-1]

        dist_variance = np.var(distances)
        dip, dip_p_value = diptest(distances)

        use_lof = (dist_variance > variance_threshold) and (dip_p_value < dip_p_value_threshold)

        if use_lof or method == 'lof':
            print("Local Outlier Factor (LOF) is used for outlier detection.")
            lof = LocalOutlierFactor(n_neighbors=k)
            y_pred = lof.fit_predict(data)
            outlier_indices.extend(data.index[y_pred == -1])

        if method == 'default':
            print("Default method for outlier detection.")
            for column in data.columns:
                if is_normal_distribution(data[column]):
                    print("normal:   ")
                    mean = data[column].mean()
                    std = data[column].std()
                    upper_limit = mean + 3 * std
                    lower_limit = mean - 3 * std
                    outliers = (data[column] > upper_limit) | (data[column] < lower_limit)
                # else:
                #     print("iqr :  ")
                #     q25 = data[column].quantile(0.25)
                #     q75 = data[column].quantile(0.75)
                #     iqr = q75 - q25
                #     upper_limit = q75 + 1.5 * iqr
                #     lower_limit = q25 - 1.5 * iqr
                #     outliers = (data[column] > upper_limit) | (data[column] < lower_limit)

                    outlier_indices.extend(data.index[outliers])

    # Remove duplicate indices and keep only valid indices
    outlier_indices = list(set(outlier_indices))
    outlier_indices = [i for i in outlier_indices if 0 <= i < len(data)]

    if outlier_indices:
        outliers = data.loc[outlier_indices]
        cleaned_data = data.drop(outlier_indices)
        cleaned_y = y.drop(outlier_indices)
    else:
        outliers = pd.DataFrame(columns=data.columns)
        cleaned_data = data
        cleaned_y = y

    return outliers, cleaned_data, cleaned_y

    
def callingfun(X, y):
    outliers, cleaned_x, cleaned_y = handle_outliers(X, y)
    print("data size: ", X.shape)
    print("y size and type: ", y.size, " and ", type(y))
    print("outlier size: ", outliers.shape)
    print("cleaned_x size: ", cleaned_x.shape)
    print("cleaned_y size: ", cleaned_y.shape)
    print("ok after getting :  ", type(cleaned_y))

    return outliers, cleaned_x, cleaned_y