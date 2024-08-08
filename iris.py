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

df = pd.read_csv('data\customer_purchase_data.csv')
y = df['PurchaseStatus']
X = df.drop('PurchaseStatus', axis=1)

print("Initial shape:  ", X.shape, " and ", y.shape, " and ", type(y))

X, y = mv.callingfunc(X, y)
outlier, X, y = out.callingfun(X, y)
X, y = fs.callingfunc(X, y)
X, y = dr.callingfunc(X, y)
X, y, strategy, model = bd.callingfunc(X, y)

print("Final shape:  ", X.shape, " and ", y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)


if(strategy == 'ensemble'):
    model.fit(X_train, Y_train)

    # Predict on the test data
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(Y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Detailed classification report
    print("Classification Report:")
    print(classification_report(Y_test, y_pred))

    # Confusion Matrix
    print("Confusion Matrix:")
    print(confusion_matrix(Y_test, y_pred))
else:
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, Y_train)
    y_pred_dt = dt_model.predict(X_test)

    accuracy_dt = accuracy_score(Y_test, y_pred_dt)
    precision_dt = precision_score(Y_test, y_pred_dt)
    recall_dt = recall_score(Y_test, y_pred_dt)
    f1_dt = f1_score(Y_test, y_pred_dt)
    cm_dt = confusion_matrix(Y_test, y_pred_dt)

    print(f'Decision Tree Classifier:\n Accuracy: {accuracy_dt}\n Precision: {precision_dt}\n Recall: {recall_dt}\n F1 Score: {f1_dt}')