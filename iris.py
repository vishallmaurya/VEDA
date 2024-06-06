import numpy as np
import pandas as pd

df = pd.read_csv('data\dirty_iris.csv')
# print(df.head(5))
# print(df.describe())
df1 = pd.read_csv('temp_file.csv')

print(len(df.columns[df.dtypes == 'object']))
print(df1.dtypes)


# print(df[df.duplicated()])
# df.drop_duplicates(inplace=True)