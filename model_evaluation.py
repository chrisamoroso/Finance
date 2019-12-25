import pandas as pd

df = pd.read_csv('NN_logs.csv')
print(df.head())

acc = df.loc[:,'acc']
print(acc.sort_values())
