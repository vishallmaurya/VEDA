import missing_value as mv
import dimensionality_reduction as dr
import feature_selection as fs
import outlier as out
import pandas as pd

df = pd.read_csv('data\dirty_iris.csv')
X = df.drop('Species', axis=1)
y = df['Species']

print("Initial shape:  ", X.shape, " and ", y.shape)

X, y = mv.callingfunc(X, y)
outlier, X, y = out.callingfun(X, y)
X, y = fs.callingfunc(X, y)
X, y = dr.callingfunc(X, y)

print("Final shape:  ", X.shape, " and ", y.shape)
