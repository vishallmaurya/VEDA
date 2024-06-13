1. describe the data (fetch important aspects from the dataset)
2. find duplicates if any then simply remove it
3. handle nan and null values using Simple Imputer library

************************ASSUMPTION IS data doesn't consist of y column********************

**********************************
Missing_value.py 

1. It doesn't handle column containing mixed values.
2. It doesn't handle column with time stamp.
3. It handle only numerical values, categorical values and temporal values.
4. Categorical values can be detected on the basis of only dtypes of columns.

***********************************


To do: check for null values in row later on!!