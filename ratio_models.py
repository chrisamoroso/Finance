import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
data = pd.read_csv('ratios.csv', index_col=['Ticker', 'Fiscal Year', 'Fiscal Period'])
print(data.head())
print(data.columns)

Y = data.loc[:,'pos_neg']
X = data.drop(columns=['pos_neg'])
X = scale(X) 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2, shuffle=False)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_model(layer1, layer2, layer3):
    model = Sequential()
    model.add(Dense(layer1, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(layer2, activation='relu'))
    model.add(Dense(layer3, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

def get_layers(numbers):
    options = [] 

    for a in numbers:
        for b in numbers:
            for c in numbers:
                options.append([a, b, c])

    return options

logger = CSVLogger('Ratio_logs.csv', append=True)

sizes=[16, 32, 64, 128]
models = []

for a, b, c in get_layers(sizes):
    model = create_model(a, b, c)
    models.append(model)

best_loss = 100

for nn in models:
    print('_'*50) 
    print(nn.summary())

    nn.fit(X_train, y_train, callbacks=[logger], batch_size=32, epochs=100)
    results = nn.evaluate(X_test, y_test) 
    print(results)
    print(nn.metrics_names)


