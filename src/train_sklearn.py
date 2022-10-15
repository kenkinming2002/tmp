#!/usr/bin/env python

from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import os

X = np.load("stage4/input.npy")
Y = np.load("stage4/output.npy")

X_scaler = preprocessing.StandardScaler().fit(X)
Y_scaler = preprocessing.StandardScaler().fit(Y)

X_scaled = X_scaler.transform(X)
Y_scaled = Y_scaler.transform(Y)

kf = KFold(n_splits=10, random_state=None,shuffle=False)
for i, (train_index, test_index) in tqdm(enumerate(kf.split(X_scaled)), desc="index", position=0, leave=False):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    Y_train, Y_test = Y_scaled[train_index], Y_scaled[test_index]

    regr = MLPRegressor(hidden_layer_sizes=(16,8,4), learning_rate_init=0.05, solver='sgd', max_iter=1000, random_state=1)
    regr.fit(X_train, Y_train.ravel())
    train_score = regr.score(X_train, Y_train)
    test_score  = regr.score(X_test,  Y_test)
    print(f"batch={i}, train_score={train_score}, test_score={test_score}\n")


#with open('results/output.txt', 'w') as f:
#    kf = KFold(n_splits=10, random_state=None,shuffle=False)
#    for i, (train_index, test_index) in tqdm(enumerate(kf.split(X_scaled)), desc="index", position=0, leave=False):
#        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
#        Y_train, Y_test = Y_scaled[train_index], Y_scaled[test_index]
#        for hidden_layer_sizes in tqdm([(16, 8, 4), (64, 64, 64)], desc="hidden_layer_sizes", position=1, leave=False):
#            for learning_rate in tqdm([0.05, 0.005], desc="learning_rate", position=2, leave=False):
#                regr = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, learning_rate_init=learning_rate, solver='sgd', max_iter=1000, random_state=1)
#                regr.fit(X_train, Y_train.ravel())
#
#                dirname = f"results/{i}/{'_'.join([str(x) for x in hidden_layer_sizes])}/{learning_rate}"
#                os.makedirs(dirname, exist_ok = True)
#
#                train_score = regr.score(X_train, Y_train)
#                test_score  = regr.score(X_test,  Y_test)
#
#                f.write(f"batch={i}, hidden_layer_sizes={hidden_layer_sizes}, learning_rate={learning_rate}, train_score={train_score}, test_score={test_score}\n")
#
#                if True:
#                    fig, ax = plt.subplots(1,1)
#                    ax.plot(np.linspace(0, np.size(regr.loss_curve_), np.size(regr.loss_curve_)), regr.loss_curve_)
#                    fig.savefig(f"{dirname}/loss_curve.png")
#                    plt.close(fig)
#
#                if True:
#                    fig, ax = plt.subplots(1,1)
#                    scatter = ax.scatter(Y_scaler.inverse_transform(regr.predict(X_train).reshape(-1,1)), Y_scaler.inverse_transform(Y_train))
#                    scatter.set_label(f"$R^2=${train_score}")
#                    ax.legend()
#                    fig.savefig(f"{dirname}/train.png")
#                    plt.close(fig)
#
#                if True:
#                    fig, ax = plt.subplots(1,1)
#                    scatter = ax.scatter(Y_scaler.inverse_transform(regr.predict(X_test).reshape(-1,1)), Y_scaler.inverse_transform(Y_test))
#                    scatter.set_label(f"$R^2=${test_score}")
#                    ax.legend()
#                    fig.savefig(f"{dirname}/test.png")
#                    plt.close(fig)
#
