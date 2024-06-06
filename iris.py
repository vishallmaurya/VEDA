import numpy as np
import pandas as pd

df = pd.read_csv('data\dirty_iris.csv')
# print(df.head(5))
# print(df.describe())
df1 = pd.read_csv('temp_file.csv')

categorical_column = list(df.columns[df.dtypes == 'object'])
numerical_column = [col for col in df.columns if col not in categorical_column]

print(f"Numerical columns name: {numerical_column}")
print(f"Categorical columns name: {categorical_column}")

print(df.shape) 
df2 = df.dropna()
print(df2.shape)

# print(df1.dtypes)


# print(df[df.duplicated()])
# df.drop_duplicates(inplace=True)