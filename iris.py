import missing_value as mv
import dimensionality_reduction as dr
import feature_selection as fs
import balance_data as bd
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
# from lightgbm import LGBMRegressor, LGBMClassifier
# from xgboost import XGBRegressor, XGBClassifier
# from catboost import CatBoostRegressor, CatBoostClassifier

df = pd.read_csv('data\campaign_data.csv')
y = df['IsSuccessful']
X = df.drop('IsSuccessful', axis=1)

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
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, Y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(Y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    print(classification_report(Y_test, y_pred))