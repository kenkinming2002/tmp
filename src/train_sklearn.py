#!/usr/bin/env python

from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import numpy as np

X = np.load("stage4/input.npy")
Y = np.load("stage4/output.npy")

X_scaler = preprocessing.StandardScaler().fit(X)
Y_scaler = preprocessing.StandardScaler().fit(Y)

X_scaled = X_scaler.transform(X)
Y_scaled = Y_scaler.transform(Y)

kf = KFold(n_splits=10, random_state=None,shuffle=False)
for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    Y_train, Y_test = Y_scaled[train_index], Y_scaled[test_index]
    regr = MLPRegressor(hidden_layer_sizes=(16, 8, 4), activation='tanh', solver='sgd', learning_rate='adaptive', max_iter=1000, random_state=1).fit(X_train, Y_train)

    print(regr.predict(X_test[:2]))
    print(Y_test[:2])
    print(regr.score(X_train, Y_train))
    print(regr.score(X_test, Y_test))


