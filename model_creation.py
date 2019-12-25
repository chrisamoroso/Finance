# Imports
from sklearn.preprocessing import scale
import itertools 
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
from sklearn.model_selection import train_test_split

df = pd.read_csv('finance_data.csv', index_col=['Ticker', 'Fiscal Year', 'Fiscal Period'])
df = df.drop(columns=['report_date', 'shifted_chg'])

Y = df.loc[:,'pos_neg'].values
df = df.drop(columns=['pos_neg'])

X = df.values
x_scale = scale(X)
print(x_scale)
X_train, X_test, y_train, y_test = train_test_split(x_scale, Y, test_size=.2, shuffle=False)

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

def create_nn(layer1, layer2, layer3):
    model = Sequential()
    model.add(Dense(layer1, activation='relu',input_dim=X.shape[1]))
    model.add(Dense(layer2, activation='relu'))
    model.add(Dense(layer3, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', # Cross-entropy
                optimizer='rmsprop', # Root Mean Square Propagation
                metrics=['accuracy']) # Accuracy performance metric
    return model

def get_layers(numbers):
    options = [] 
    for layer1 in numbers:
        for layer2 in numbers:
            for layer3 in numbers:
                options.append([layer1, layer2, layer3])

    return options


logger = CSVLogger('NN_logs.csv', append=True)

sizes = [32, 64, 128]
models = []
for layer1, layer2, layer3 in get_layers(sizes):
    model = create_nn(layer1,layer2,layer3)
    models.append(model) 
models.pop(0)
best_loss = 1
for nn in models:
    print('New Model')
    print('----------------------------------')
    print(nn.summary())
    nn.fit(X_train, y_train, callbacks=[logger], batch_size=32, epochs=100)
    results = nn.evaluate(X_test, y_test)
    print(results)
    print(nn.metrics_names)
    if results[0] < best_loss:
        #Save model
        print('new best results')
        print(nn.summary)
        print(nn.metrics_names)
        print(results)
        best_loss = results[0]

