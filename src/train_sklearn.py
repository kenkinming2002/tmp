#!/usr/bin/env python

from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

import numpy as np

X = np.load("stage4/input.npy")
Y = np.load("stage4/output.npy")

X_scaler = preprocessing.StandardScaler().fit(X)
Y_scaler = preprocessing.StandardScaler().fit(Y)

X_scaled = X_scaler.transform(X)
Y_scaled = Y_scaler.transform(Y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_scaled, random_state=1)

regr = MLPRegressor(hidden_layer_sizes=(32, 32, 32), activation='tanh', solver='sgd', learning_rate = 'adaptive', max_iter=1000, random_state=1).fit(X_train, y_train)
print(regr.predict(X_test[:2]))
print(regr.score(X_train, y_train))
print(regr.score(X_test, y_test))


