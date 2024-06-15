import pandas as pd
import numpy as np
from scipy.stats import shapiro, kstest, anderson, jarque_bera, skew, kurtosis

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

# Example usage:

data = pd.read_csv('data\placement.csv')
result = is_normal_distribution(data['placement_exam_marks'], tests=['jarque-bera'])
print(f"Is the data normally distributed? {result}")
