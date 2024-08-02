import missing_value as mv
import dimensionality_reduction as dr
import feature_selection as fs
import outlier as out
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('data\\nearest-earth-objects(1910-2024).csv')
X = df.drop('is_hazardous', axis=1)
y=df['is_hazardous']

print("Initial shape:  ", X.shape, " and ", y.shape, " and ", type(y))

X, y = mv.callingfunc(X, y)
outlier, X, y = out.callingfun(X, y)
X, y = fs.callingfunc(X, y)
X, y = dr.callingfunc(X, y)

print("Final shape:  ", X.shape, " and ", y.shape)

x_train,x_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=42,stratify=y)

model=RandomForestClassifier()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print(accuracy_score(y_pred,y_test))