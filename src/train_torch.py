#!/usr/bin/env python

from tqdm import tqdm

from train_base import *

import sys
import math

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(6, 32),
            nn.Tanh(),
            nn.Linear(32, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.stack(x)

def train(model, epochs, batch_size, learning_rate, dataset):
    dataloader = DataLoader(dataset, batch_size = batch_size)

    size = len(dataloader.dataset)

    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(title, model, batch_size, dataset):
    dataloader = DataLoader(dataset, batch_size = batch_size)

    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    loss_fn = nn.L1Loss()

    ss_res = 0.0
    ss_tot = 0.0

    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            ss_res += (y - pred).square().sum()
            ss_tot += (y - y.mean()).square().sum()

    r2 = 1 - ss_res / ss_tot
    print(f"Testing({title} dataset):ss_res={ss_res}, ss_tot={ss_tot}, R2={r2}")

    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


sub_count = int(count / 10)
for i in range(10):
    # Dataset
    training_input  = np.concatenate((all_input [:i*sub_count], all_input [(i+1)* sub_count:]))
    training_output = np.concatenate((all_output[:i*sub_count], all_output[(i+1)* sub_count:]))

    testing_input  = all_input [i * sub_count: (i+1)*sub_count]
    testing_output = all_output[i * sub_count: (i+1)*sub_count]

    # Shuffle
    tmp = np.random.permutation(training_input.shape[0])
    training_input  = training_input[tmp]
    training_output = training_output[tmp]

    tmp = np.random.permutation(testing_input.shape[0])
    testing_input  = testing_input[tmp]
    testing_output = testing_output[tmp]

    training_dataset = TensorDataset(torch.Tensor(training_input), torch.Tensor(training_output))
    testing_dataset  = TensorDataset(torch.Tensor(testing_input),  torch.Tensor(testing_output))

    # Model
    model = Model()

    test("testing",  model, 128, testing_dataset)
    test("training", model, 128, training_dataset)

    while True:
        train(model, 1, 4, 0.05, training_dataset)
        test("testing",  model, 128, testing_dataset)
        test("training", model, 128, training_dataset)

    test("testing",  model, 128, testing_dataset)
    test("training", model, 128, training_dataset)
    sys.exit(1)


