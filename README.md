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

```bash
from veda_lib import Veda
```