#!/usr/bin/env python

from tqdm import tqdm

import sys
import math

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt


#################################### Load and prepare the data set #############################################################

# Load
with open('dataset', 'rb') as f:
    all_input  = np.load(f, allow_pickle = True)
    all_output = np.load(f, allow_pickle = True)

#all_input  = np.random.uniform(-0.3, 0.3, (50000, 6))
#all_output = np.sum(all_input, 1, keepdims = True)

count = all_output.shape[0]

print(all_input.shape)
print(all_output.shape)

print(np.count_nonzero(np.isnan(all_input)))
print(np.count_nonzero(np.isnan(all_output)))

# Shuffle
#tmp = np.random.permutation(count)
#all_input  = all_input[tmp]
#all_output = all_output[tmp]

# Debug
print(f"Before normalization - Input:  mean = {np.mean(all_input,  axis=0)}, std = {np.std(all_input,  axis=0)}")
print(f"Before normalization - Output: mean = {np.mean(all_output, axis=0)}, std = {np.std(all_output, axis=0)}")

# Normalize
#all_input  = (all_input  - np.mean(all_input,  axis=0)) / np.std(all_input,  axis=0)
#all_input  = (all_input  - np.amin(all_input,  axis=0)) / (np.amax(all_input,  axis=0) - np.amin(all_input,  axis=0)) * 2.0 - 1.0
#all_input = (all_input - np.median(all_input, axis=0)) / np.std(all_input, axis=0)
all_input = (all_input - np.median(all_input, axis=0)) / (np.amax(all_input,  axis=0) - np.amin(all_input,  axis=0))

#all_output = (all_output - np.mean(all_output, axis=0)) / np.std(all_output, axis=0)
#all_output = (all_output - np.amin(all_output, axis=0)) / (np.amax(all_output, axis=0) - np.amin(all_output, axis=0)) * 2.0 - 1.0
#all_output = (all_output - np.median(all_output, axis=0)) / np.std(all_output, axis=0)
all_output = (all_output - np.median(all_output, axis=0)) / (np.amax(all_output, axis=0) - np.amin(all_output, axis=0))

fig, axs = plt.subplots(1, 7)

axs[0].hist(all_input[...,0], bins = 100)
axs[1].hist(all_input[...,1], bins = 100)
axs[2].hist(all_input[...,2], bins = 100)
axs[3].hist(all_input[...,3], bins = 100)
axs[4].hist(all_input[...,4], bins = 100)
axs[5].hist(all_input[...,5], bins = 100)

axs[6].hist(all_output[...,0], bins = 100)

axs[0].set_title("AOD")
axs[1].set_title("PBLH")
axs[2].set_title("PS")
axs[3].set_title("QV10M")
axs[4].set_title("T2M")
axs[5].set_title("WS")

axs[6].set_title("PM25")

plt.show()

# Debug
print(f"After normalization - Input:  mean = {np.mean(all_input,  axis=0)}, std = {np.std(all_input,  axis=0)}")
print(f"After normalization - Output: mean = {np.mean(all_output, axis=0)}, std = {np.std(all_output, axis=0)}")

#################################### Optimization #############################################################

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(6, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
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


def test(model, batch_size, dataset):
    dataloader = DataLoader(dataset, batch_size = batch_size)

    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    loss_fn = nn.L1Loss()

    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


sub_count = int(count / 10)
for i in range(10):
    # Dataset
    training_input  = np.concatenate((all_input [:i*sub_count], all_input [(i+1)* sub_count:]))
    training_output = np.concatenate((all_output[:i*sub_count], all_output[(i+1)* sub_count:]))

    testing_input  = all_input [i * sub_count: (i+1)*sub_count]
    testing_output = all_output[i * sub_count: (i+1)*sub_count]

    training_dataset = TensorDataset(torch.Tensor(training_input), torch.Tensor(training_output))
    testing_dataset  = TensorDataset(torch.Tensor(testing_input),  torch.Tensor(testing_output))

    # Model
    model = Model()

    print("Testing on testing data")
    test(model, 128, testing_dataset)
    print("Testing on training data")
    test(model, 128, training_dataset)

    for i in range(100):
        train(model, 1, 32, 0.05, training_dataset)

        print("Testing on testing data")
        test(model, 128, testing_dataset)
        print("Testing on training data")
        test(model, 128, training_dataset)

    print("Testing on testing data")
    test(model, 128, testing_dataset)
    print("Testing on training data")
    test(model, 128, training_dataset)
    sys.exit(1)


