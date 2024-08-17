import missing_value as mv
import dimensionality_reduction as dr
import feature_selection as fs
import balance_data as bd
import outlier as out
import pandas as pd
import numpy as np

df = pd.read_csv('data\\train1.csv')
# df2 = pd.read_csv('data\\test1.csv')

# df = pd.concat([df, df2], ignore_index=True)

X = df.drop(['Loan Status'],axis=1)
y = df['Loan Status']

print("Initial shape:  ", X.shape, " and ", y.shape, " and ", type(y))

X, y = mv.callingfunc(X, y)
outlier, X, y = out.callingfun(X, y)
X, y = fs.callingfunc(X, y)
X, y = dr.callingfunc(X, y)
X, y, strategy, model = bd.callingfunc(X, y, classification=True)

print("Final shape:  ", X.shape, " and ", y.shape)
print("Final shape:  ", X.isna().sum(), " and ", y.isna().sum())

new_df = X.copy()  # Make a copy of X to avoid any unwanted changes
new_df['target'] = y.values 
print(new_df.isna().sum())

new_df.to_csv('cleaned_data\\delloite.csv', index=False)