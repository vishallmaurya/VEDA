# veda_lib

**A Python library designed to streamline the transition from raw data to machine learning models.**  
veda_lib automates and simplifies data preprocessing, cleaning, and balancing, addressing the time-consuming and complex aspects of these tasks to provide clean, ready-to-use data for your models.

********************************************************

## Installation

First, install `veda_lib` using pip:

```bash
pip install veda_lib
```

**************************************

## How to use?

After installing `veda_lib`, import it into your project and start utilizing its modules to prepare your data. Below is a summary of the key functionalities provided by each module:

**1. Preprocessor Module**

- Functions:
   - Removing null values
   - Handling duplicates
   - Imputing missing values with appropriate methods
- Usage: Ideal for initial data cleaning and preprocessing steps.

- Parameters:

   - **keep** *(str/bool, default=`'first'`)*  
   How to keep duplicates. Options: `['first', 'last', False]`.

   - **min_cat_percent** *(float, default=`5`)*  
   Convert column into categorical if % of unique values < threshold.

   - **datalosspercent** *(float, default=`10`)*  
   Maximum acceptable % of data loss during cleaning.

   - **min_var** *(float, default=`0.04`)*  
   Row deletion threshold. Columns with missing proportion > `min_var` are ignored.

   - **min_col_threshold** *(float, default=`0.65`)*  
   Column deletion threshold. Drop columns with missing proportion > threshold.

   - **var_diff** *(float, default=`0.05`)*  
   Maximum allowable variance change (numerical imputation).

   - **mod_diff** *(float, default=`0.05`)*  
   Threshold for mode dominance (categorical imputation).

   - **numerical_column** *(list/None, default=`None`)*  
   List of numerical column names (if not auto-detected).

   - **categorical_column** *(list/None, default=`None`)*  
   List of categorical column names (if not auto-detected).

   - **temporal_column** *(list/None, default=`None`)*  
   List of temporal column names (if any).

   - **temporal_type** *(str, default=`'interpolate'`)*  
   Strategy for temporal imputation. Options: `['bfill', 'ffill', 'interpolate']`.

   - **n_neighbors** *(int, default=`5`)*  
   Number of neighbors for multivariate imputation (KNN-based).

   - **label_encoding_type** *(str, default=`'onehot'`)*  
   Encoding strategy for categorical features. Options: `['onehot', 'labelencode']`.

*******************************************************

**2. OutlierHandler Module**
- Functions:
   - Handling outliers by either removing or capping them
   - Customizable based on the nature of your data
- Usage: Useful for managing data skewness and ensuring robust model performance.

- Parameters:

   - **tests** *(list, default=`['skew-kurtosis']`)*
   Test to check whether the data is having normal distribution or not. Options: 
      - *shapiro: Tests the null hypothesis that the data was drawn from a normal distribution.*
      - *skew-kurtosis: skewness measures asymmetry in the data, normal distribution has skewness app. 0 and kurtosis measures "peakedness", normal distribution has kurtosis app.*
      - *kstest: Compares the sample distribution with a theoretical normal distribution*
      - *Anderson: Checks how well data fits a normal distribution, focusing more on the tails*
      - *jarque-bera: Checks if skewness and kurtosis match those of a normal distribution.*
   
   - **method** *(str, default=`'default'`)*
   Outliers detection stratedy. Options:
      - *default: Adaptive pipeline (Dip Test + DBSCAN | Isolation Forest | LOF | Normal Rule)*
      - *isolation forest: Always uses isolation forest*
      - *lof: Always uses local outlier factor*

   - **handle** *(str, default=`'capping'`)*
   Strategy for handling detected outliers. Options:
      - *capping: Replace values beyond 3*var limits with boundary values*
      - *trimming: Drop rows with outliers.*
      - *winsorization: Clip values at limits.*

   - **minlen** *(int, defualt=`5000`)*
   Minimum dataset size above which Shapiro test is applied.
   
   - **skew_thresh** *(float, default=`1`)*
   Absolute skewness threshold. Values greater than this indicate non-normal distribution.

   - **kurt_thresh** *(float, default=`1`)*
   Absolute deviation from kurtosis=3 (normal distribution). Values greater than this indicate non-normal distribution.

*******************************************************

**3. FeatureSelector Module**
- Functions:
   - Selecting important features from the dataset
   - Tailored selection based on the nature of the data
- Usage: Helps in reducing dimensionality and focusing on the most impactful features.

- Parameters:

   - **percentile** *(float, default=`90`)*
   Percentile threshold (0–100) for selecting features most correlated with the target variable. Higher values select fewer features with stronger correlations.
   
   - **threshold** *(float, default=`0.9`)*
   Cumulative mutual information threshold (0–1) that determines the optimal number of features to select. A higher threshold selects more features.

   - **cv** *(int, default=`5`)*
   Number of cross-validation folds for selecting the best Lasso regularization strength (alpha). Must be a positive integer.

*******************************************************

**4. DimensionReducer Module**
- Functions:
   - Reducing data dimensionality using appropriate techniques
- Usage: Crucial for addressing the curse of dimensionality and improving model efficiency.

- Parameters:

   - **variance_threshold** *(float, default=`0.95`)*
   Fraction of variance to preserve during PCA/autoencoder training.

   - **prioritize_reproducibility** *(bool, default=`True`)*
   Ensures deterministic results by fixing random seeds.

   - **min_neighbors** *(int, default=`5`)*
   Minimum number of neighbors to controls local structure preservation.

   - **max_neighbors** *(int, default=`50`)*
   Maximum number of neighbors to prevents over-smoothing of high-dimensional manifolds.
   
   - **min_dim** *(int, default=`10`)*
   Minimum encoding dimension for Autoencoders.

   - **max_dim** *(int, default=`100`)*
   Maximum encoding dimension for Autoencoders.

   - **hidden_layers** *(int, default=`1`)*
   Number of hidden layers in Autoencoder.

   - **optimizer** *(str, default=`adam`)*
   Optimizer used for training Autoencoders.

   - **loss** *(str, default=`mean_squared_error`)*
   Loss function for Autoencoder reconstruction.

   - **min_epochs** *(int, default=`20`)*
   Minimum number of epochs for Autoencoder training.

   - **max_epochs** *(int, default=`100`)*
   Maximum epochs allowed for training Autoencoders.

   - **min_batch_size** *(int, default=`32`)*
   Smallest batch size for Autoencoder training.

   - **max_batch_size** *(int, default=`256`)*
   Largest batch size allowed for Autoencoder training.

*******************************************************

**5. BalanceData Module**
- Functions:
   - Balancing class distribution in imbalanced datasets
   - Methods chosen based on data characteristics
- Usage: Essential for improving model fairness and performance on imbalanced datasets.

- Parameters: 

   - **threshold** *(float, 0.5)*
   Minimum acceptable ratio of minority to majority class. If the imbalance ratio is greater than or equal to this threshold, no resampling is performed.

   - **classification** *(bool, None)*
   Whether the task is classification or not. Options: `[True, False]`

*******************************************************

**6. Veda Module**
- Functions:
   - Integrates all the above functionalities into a single pipeline
- Usage: Pass your raw data through this module to perform comprehensive EDA and get fully preprocessed, cleaned, and balanced data ready for model training.

- Parameters:

   - **classification** *(bool, None)*
   Whether the task is classification or not. Options: `[True, False]`

*******************************************************

## Importing

- Here is an example of importing Veda from veda_lib.Veda, here set classification to True if the problem is classification otherwise set to False.
```bash
from veda_lib import Veda
```
```bash
eda = Veda.Veda(classification=True)
X, y, outliers, strategy, model = eda.fit_transform(X, y)
```


- Here is an example of importing DataPreprocessor from veda_lib.Preprocessor, using default values of parameters
```bash
from veda_lib import Preprocessor
```
```bash
preprocessor = Preprocessor.DataPreprocessor()
X, y = preprocessor.fit_transform(X, y)
```


- Here is an example of importing OutlierPreprocessor from veda_lib.OutlierHandler, using default values of parameters.
```bash
from veda_lib import OutlierHandler
```
```bash
outlier_preprocessor = OutlierHandler.OutlierPreprocessor()
X, y = outlier_preprocessor.fit_transform(X, y)
```


- Here is an example of importing FeatureSelection from veda_lib.FeatureSelector, using default values of parameters.
```bash
from veda_lib import FeatureSelector
```
```bash
selector = FeatureSelector.FeatureSelection()
X, y = selector.fit_transform(X, y)
```


- Here is an example of importing DimensionReducer from veda_lib.DimensionReducer, using default values of parameters.
```bash
from veda_lib import DimensionReducer
```
```bash
reducer = DimensionReducer.DimensionReducer()
X, y = reducer.fit_transform(X, y)
```


- Here is an example of importing AdaptiveBalancer from veda_lib.BalanceData, using default values of parameters.
```bash
from veda_lib import BalanceData
```
```bash
balancer = BalanceData.AdaptiveBalancer(classification=True)
X, y, strategy, model = balancer.fit_transform(X, y)
```

**************************************************************** 

## Contributing

I welcome contributions to `veda_lib`! If you have a bug report, feature suggestion, or want to contribute code, please open an issue or pull request on GitHub.

*************************************************************

## License

`veda_lib` is licensed under the Apache License Version 2.0. See the [LICENSE](https://github.com/vishallmaurya/VEDA?tab=Apache-2.0-1-ov-file) file for more details.

