import missing_value as mv
import dimensionality_reduction as dr
import feature_selection as fs
import outlier as out
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, RandomForestClassifier, VotingClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

df = pd.read_csv('data\\alzheimers_disease_data.csv')
y = df['Diagnosis']
X = df.drop('Diagnosis', axis=1)

print("Initial shape:  ", X.shape, " and ", y.shape, " and ", type(y))

X, y = mv.callingfunc(X, y)
outlier, X, y = out.callingfun(X, y)
X, y = fs.callingfunc(X, y)
X, y = dr.callingfunc(X, y)

print("Final shape:  ", X.shape, " and ", y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, Y_train)

print(f"Best parameters: {grid_search.best_params_}")

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

print(classification_report(Y_test, y_pred))

print(confusion_matrix(Y_test,y_pred))