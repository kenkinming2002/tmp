#!/usr/bin/env python

from tqdm import tqdm

from load import *

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

def train(model, epochs, batch_size, learning_rate, training_input, training_output):
    training_input  = normalize_input(training_input)
    training_output = normalize_output(training_output)

    dataset = TensorDataset(torch.Tensor(training_input), torch.Tensor(training_output))
    dataloader = DataLoader(dataset, batch_size = batch_size)

    size = len(dataloader.dataset)

    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for k in range(epochs):
        for batch, (X, y) in enumerate(dataloader):
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 500 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(title, model, batch_size, testing_input, testing_output):
    testing_input  = normalize_input(testing_input)
    testing_output = normalize_output(testing_output)

    dataset = TensorDataset(torch.Tensor(testing_input), torch.Tensor(testing_output))
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

count = all_input.shape[0]
sub_count = int(count / 10)
for i in range(10):
    # Dataset
    training_input  = np.concatenate((all_input [:i*sub_count], all_input [(i+1)* sub_count:]))
    training_output = np.concatenate((all_output[:i*sub_count], all_output[(i+1)* sub_count:]))

    testing_input  = all_input [i * sub_count: (i+1)*sub_count]
    testing_output = all_output[i * sub_count: (i+1)*sub_count]

    # Shuffle
    testing_input,  testing_output  = shuffle(testing_input,  testing_output)
    training_input, training_output = shuffle(training_input, training_output)

    # Model
    model = Model()

    test("testing",  model, 128, testing_input, testing_output)
    test("training", model, 128, training_input, training_output)

    train(model, 10, 4, 0.001, training_input, training_output)

    test("testing",  model, 128, testing_input, testing_output)
    test("training", model, 128, training_input, training_output)


