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

**2. OutlierHandler Module**
- Functions:
   - Handling outliers by either removing or capping them
   - Customizable based on the nature of your data
- Usage: Useful for managing data skewness and ensuring robust model performance.

**3. FeatureSelector Module**
- Functions:
   - Selecting important features from the dataset
   - Tailored selection based on the nature of the data
- Usage: Helps in reducing dimensionality and focusing on the most impactful features.

**4. DimensionReducer Module**
- Functions:
   - Reducing data dimensionality using appropriate techniques
- Usage: Crucial for addressing the curse of dimensionality and improving model efficiency.

**5. BalanceData Module**
- Functions:
   - Balancing class distribution in imbalanced datasets
   - Methods chosen based on data characteristics
- Usage: Essential for improving model fairness and performance on imbalanced datasets.

**6. Veda Module**
- Functions:
   - Integrates all the above functionalities into a single pipeline
- Usage: Pass your raw data through this module to perform comprehensive EDA and get fully preprocessed, cleaned, and balanced data ready for model training.

*******************************************************

## Importing

- Here is an example of importing Veda from veda_lib.Veda, here set classification to True if the problem is classification otherwise set to False.
```bash
from veda_lib import Veda
```
```bash
eda = Veda.Veda(classification=True)
eda.fit_transform(X, Y)
```


- Here is an example of importing DataPreprocessor from veda_lib.Preprocessor, using default values of parameters
```bash
from veda_lib import Preprocessor
```
```bash
preprocessor = Preprocessor.DataPreprocessor()
X, y = preprocessor.fit_transform(X, Y)
```


- Here is an example of importing OutlierPreprocessor from veda_lib.OutlierHandler, using default values of parameters.
```bash
from veda_lib import OutlierHandler
```
```bash
outlier_preprocessor = OutlierHandler.OutlierPreprocessor()
X, y = outlier_preprocessor.fit_transform(X, Y)
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

