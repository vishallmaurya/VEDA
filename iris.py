import numpy as np
import pandas as pd

df = pd.read_csv('dirty_iris.csv')
# print(df.head(5))
# print(df.describe())

print(df[df.duplicated()])
df.drop_duplicates(inplace=True)

df