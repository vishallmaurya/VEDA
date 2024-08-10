# import missing_value as mv
# import dimensionality_reduction as dr
# import feature_selection as fs
# import balance_data as bd
# import outlier as out
# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score,classification_report, confusion_matrix
# from sklearn.linear_model import LogisticRegression
# import numpy as np
# from sklearn.model_selection import StratifiedKFold
# from sklearn.tree import DecisionTreeClassifier 
# from sklearn.ensemble import RandomForestRegressor, VotingRegressor, RandomForestClassifier, VotingClassifier
# from sklearn.ensemble import (ExtraTreesClassifier, 
#                               GradientBoostingClassifier, 
#                               HistGradientBoostingClassifier)
# from lightgbm import LGBMRegressor, LGBMClassifier
# from xgboost import XGBRegressor, XGBClassifier
# from catboost import CatBoostRegressor, CatBoostClassifier
# from sklearn.metrics import (balanced_accuracy_score as bas, 
#                              confusion_matrix)

# df = pd.read_csv('data\online_course_engagement_data.csv')

# X = df.drop(columns=['CourseCompletion'])
# y = df['CourseCompletion']

# print("Initial shape:  ", X.shape, " and ", y.shape, " and ", type(y))

# X, y = mv.callingfunc(X, y)
# outlier, X, y = out.callingfun(X, y)
# X, y = fs.callingfunc(X, y)
# X, y = dr.callingfunc(X, y)
# X, y, strategy, model = bd.callingfunc(X, y)

# print("Final shape:  ", X.shape, " and ", y.shape)

# X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# if(strategy == 'ensemble'):
#     model.fit(X_train, Y_train)

#     # Predict on the test data
#     y_pred = model.predict(X_test)

#     # Evaluate the model
#     accuracy = accuracy_score(Y_test, y_pred)
#     print(f"Accuracy: {accuracy}")

#     # Detailed classification report
#     print("Classification Report:")
#     print(classification_report(Y_test, y_pred))

#     # Confusion Matrix
#     print("Confusion Matrix:")
#     print(confusion_matrix(Y_test, y_pred))
# else:
#     log_reg = GradientBoostingClassifier(learning_rate=0.2, max_depth = 5, min_samples_split=3, n_estimators=300, subsample=0.8)
#     log_reg.fit(X_train, Y_train)

#     # Predict on the test set
#     y_pred = log_reg.predict(X_test)

#     # Calculate accuracy, F1 score, recall
#     accuracy = accuracy_score(Y_test, y_pred)
#     f1 = f1_score(Y_test, y_pred)
#     recall = recall_score(Y_test, y_pred)
#     conf_matrix = confusion_matrix(Y_test, y_pred)
#     print('Classification Report:')
#     print(classification_report(Y_test, y_pred))
#     # Display the results
#     print(f'Accuracy: {accuracy:.4f}')
#     print(f'F1 Score: {f1:.4f}')
#     print(f'Recall: {recall:.4f}')
#     print('Confusion Matrix:')
#     print(conf_matrix)

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

df = pd.read_csv('data\online_course_engagement_data.csv')

X = df.drop(columns=['CourseCompletion'])
y = df['CourseCompletion']

print("Initial shape:  ", X.shape, " and ", y.shape, " and ", type(y))

X, y = mv.callingfunc(X, y)
outlier, X, y = out.callingfun(X, y)
X, y = fs.callingfunc(X, y)
X, y = dr.callingfunc(X, y)
X, y, strategy, model = bd.callingfunc(X, y)

print("Final shape:  ", X.shape, " and ", y.shape)

new_df = X
new_df['target'] = y

new_df.to_csv('handle_this_data.csv')

# X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# if(strategy == 'ensemble'):
#     model.fit(X_train, Y_train)

#     # Predict on the test data
#     y_pred = model.predict(X_test)

#     # Evaluate the model
#     accuracy = accuracy_score(Y_test, y_pred)
#     print(f"Accuracy: {accuracy}")

#     # Detailed classification report
#     print("Classification Report:")
#     print(classification_report(Y_test, y_pred))

#     # Confusion Matrix
#     print("Confusion Matrix:")
#     print(confusion_matrix(Y_test, y_pred))
# else:
#     # Hyperparameter tuning with GridSearchCV
#     param_grid = {
#         'n_estimators': [100, 200, 300],
#         'learning_rate': [0.01, 0.1, 0.2],
#         'max_depth': [3, 4, 5],
#         'subsample': [0.8, 0.9, 1.0],
#         'min_samples_split': [2, 3, 4]
#     }

#     grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
#     grid_search.fit(X_train, Y_train)

#     # Best model from GridSearchCV
#     best_model = grid_search.best_estimator_
#     print(f'Best parameters found: {grid_search.best_params_}')

#     # Evaluating the best model
#     y_pred = best_model.predict(X_test)
#     accuracy = accuracy_score(Y_test, y_pred)
#     print(f'Accuracy: {accuracy}')
#     print('Classification Report:')
#     print(classification_report(Y_test, y_pred))
#     print('Confusion Matrix:')
#     print(confusion_matrix(Y_test, y_pred))