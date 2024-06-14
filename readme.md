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

























Exploratory Data Analysis (EDA) is a crucial step in the data analysis process. It involves examining the dataset to understand its structure, detect anomalies, and uncover patterns, relationships, and insights. Here are the detailed steps involved in EDA:

1. Data Collection
Before starting EDA, ensure that you have collected the data from various sources such as databases, web scraping, APIs, or other files like CSV, Excel, etc.

2. Data Cleaning
a. Handle Missing Values
    1. Identify missing values: Use methods like isnull() or info() in Pandas to check for missing values.
    2. Handling strategies:
       2.1 Remove missing values: Use dropna().
       2.2 Impute missing values: Fill missing values using fillna(), mean, median, mode, or more complex methods like K-Nearest Neighbors.

b. Handle Outliers
    1. Identify outliers: Use methods like box plots, z-scores, or IQR.
    2. Handling strategies:
        2.1 Remove outliers: Filter out outliers.
        2.2 Transform data: Apply transformations like log or square root.
        2.3 Cap values: Cap extreme values to a certain threshold.

c. Handle Duplicates
    1. Identify duplicates: Use duplicated().
    2. Remove duplicates: Use drop_duplicates().


3. Data Transformation
a. Data Type Conversion
    Convert data types: Use methods like astype() to ensure columns have appropriate data types.
b. Feature Engineering
    1. Create new features: Based on domain knowledge, create new features that might be helpful for analysis.
    2. Binning: Discretize continuous variables into bins or categories.


4. Data Visualization
a. Univariate Analysis
    1. Visualize single variable distributions: Use histograms, box plots, and density plots.
b. Bivariate Analysis
    1. Visualize relationships between two variables: Use scatter plots, bar plots, and correlation matrices.
c. Multivariate Analysis
    1. Visualize relationships among multiple variables: Use pair plots, heatmaps, and 3D scatter plots.


5. Summary Statistics
a. Descriptive Statistics
    1. Calculate summary statistics: Use describe() to get mean, median, standard deviation, etc.
b. Grouping and Aggregation
    1. Group data: Use groupby() to calculate aggregate statistics for different groups.


6. Correlation Analysis
    1. Calculate correlation matrix: Use corr() to find correlations between variables.
    2. Visualize correlations: Use heatmaps to visualize correlation matrices


7. Hypothesis Testing
    1. Formulate hypotheses: Based on initial observations, formulate hypotheses about the data.
    2. Test hypotheses: Use statistical tests like t-tests, chi-square tests, or ANOVA to validate hypotheses.