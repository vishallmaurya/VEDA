import missing_value as mv
import dimensionality_reduction as dr
import feature_selection as fs
import balance_data as bd
import outlier as out
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score,classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, RandomForestClassifier, VotingClassifier
from sklearn.ensemble import (ExtraTreesClassifier, 
                              GradientBoostingClassifier, 
                              HistGradientBoostingClassifier)
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.metrics import (balanced_accuracy_score as bas, 
                             confusion_matrix)

import Preprocessor as preprocess
import OutlierHandler as outlierhandler
import FeatureSelector as featureselector
import DimensionReducer
import BalanceData as balancedata
import missing_value as mv
import feature_selection as fs

df = pd.read_csv('data\\train1.csv')
# df2 = pd.read_csv('data\\test1.csv')

# df = pd.concat([df, df2], ignore_index=True)

X = df.drop(['Loan Status'],axis=1)
y = df['Loan Status']

print("Initial shape:  ", X.shape, " and ", y.shape, " and ", type(y))

preprocess_obj = preprocess.DataPreprocessor()
X, y = preprocess_obj.fit_transform(X, y)

# out = outlierhandler.OutlierPreprocessor()
# outlier, X, y  = out.fit_transform(X, y)

# featuresel = featureselector.FeatureSelectionPipeline()
# X, y = featuresel.fit_transform(X, y)


# dimred = DimensionReducer.DimensionReducer()
# X, y = dimred.fit_transform(X, y)

bdata = balancedata.AdaptiveBalancer(classification=True)
X, y, strategy, model = bdata.fit_transform(X, y)

print("Final shape:  ", X.shape, " and ", y.shape)
print("Final shape:  ", X.isna().sum(), " and ", y.isna().sum())

new_df = X.copy()  # Make a copy of X to avoid any unwanted changes
new_df['target'] = y.values 
print(new_df.isna().sum())

# new_df.to_csv('cleaned_data\\delloite.csv', index=False)