import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score

#df = pd.read_csv('finance_data.csv', index_col=['Ticker', 'Fiscal Year', 'Fiscal Period'])
df = df.drop(columns=['report_date', 'shifted_chg'])
Y = df.loc[:,'pos_neg'].values
df = df.drop(columns=['pos_neg'])

X = df.values
x_scale = scale(X)
print(x_scale)
data = pd.read_csv('ratios.csv', index_col=['Ticker', 'Fiscal Year', 'Fiscal Period'])
print(data.head())
print(data.columns)

Y = data.loc[:,'pos_neg']
X = data.drop(columns=['pos_neg'])
X = scale(X) 

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2, shuffle=False)
h = .02  # step size in the mesh 


Z = model.predict(X_test)
acc = accuracy_score(y_test, Z)
print(acc)
print(y_test[0:10])
print(Z[0:10])

