#!/usr/bin/env python

from tqdm import tqdm

import sys
import math

import numpy as np
import matplotlib.pyplot as plt

class DenseLayer:
    def __init__(self, input_size, output_size):
        self.weight = np.random.normal(0, math.sqrt(2.0/(input_size + output_size)), (input_size, output_size))
        self.bias   = np.random.normal(0, 0.1, output_size)

    def forward(self, input_value):
        self.saved_input = input_value
        return np.matmul(self.saved_input, self.weight) + self.bias

    def backward(self, output_gradient):
        input_gradient       = np.matmul(output_gradient, self.weight.transpose())
        self.weight_gradient = np.matmul(self.saved_input.transpose(), output_gradient)
        self.bias_gradient   = np.sum(output_gradient, 0)
        return input_gradient

    def train(self, learning_rate):
        self.weight -= learning_rate * self.weight_gradient
        self.bias   -= learning_rate * self.bias_gradient

class ActivationLayer:
    def __init__(self):
        pass

    def forward(self, input_value):
        self.saved_input = input_value
        return np.tanh(input_value)

    def backward(self, output_gradient):
        return output_gradient / np.square(np.cosh(self.saved_input))

    def train(self, learning_rate):
        pass

class Layer:
    def __init__(self, input_size, output_size):
        self.dense      = DenseLayer(input_size, output_size)
        self.activation = ActivationLayer()

    def forward(self, input_value):
        return self.activation.forward(self.dense.forward(input_value))

    def backward(self, output_gradient):
        return self.dense.backward(self.activation.backward(output_gradient))

    def train(self, learning_rate):
        self.dense.train(learning_rate)
        self.activation.train(learning_rate)

class Model:
    def __init__(self):
        self.layer1 = Layer(6, 16)
        self.layer2 = Layer(16, 8)
        self.layer3 = Layer(8, 4)
        self.layer4 = Layer(4, 1)

    def forward(self, input_value):
        output_value = input_value
        output_value = self.layer1.forward(output_value)
        output_value = self.layer2.forward(output_value)
        output_value = self.layer3.forward(output_value)
        output_value = self.layer4.forward(output_value)
        return output_value

    def backward(self, output_gradient):
        input_gradient = output_gradient
        input_gradient = self.layer4.backward(input_gradient)
        input_gradient = self.layer3.backward(input_gradient)
        input_gradient = self.layer2.backward(input_gradient)
        input_gradient = self.layer1.backward(input_gradient)
        return input_gradient

    def train(self, learning_rate):
        self.layer1.train(learning_rate)
        self.layer2.train(learning_rate)
        self.layer3.train(learning_rate)
        self.layer4.train(learning_rate)

def train(model, epoch, batch_size, learning_rate, training_input, training_output):
    size        = len(training_input)
    batch_count = int(size / batch_size)
    for k in range(epoch):
        for i in tqdm(range(batch_count), desc=f"Training(epoch={k})"):
            batch_input  = training_input[i*batch_size:(i+1)*batch_size]
            batch_output = training_output[i*batch_size:(i+1)*batch_size]

            batch_prediction = model.forward(batch_input)

            batch_output_gradient = (batch_prediction - batch_output) * 2
            batch_input_gradient  = model.backward(batch_output_gradient)

            model.train(learning_rate)

def test(model, testing_input, testing_output):
    prediction = model.forward(testing_input)

    real_testing_input  = (testing_input  * all_input_std)  + all_input_mean
    real_testing_output = (testing_output * all_output_std) + all_output_mean
    real_prediction     = (prediction     * all_output_std) + all_output_mean

    for i in range(5):
        print(f"Example: input={real_testing_input[i]}, prediction = {real_prediction[i]}, output={real_testing_output[i]}")

    ss_res = np.square(real_testing_output - real_prediction).sum()
    ss_tot = np.square(real_testing_output - real_testing_output.mean()).sum()
    r2 = 1 - ss_res / ss_tot
    print(f"Testing:ss_res={ss_res}, ss_tot={ss_tot}, R2={r2}")

# Load
with open('dataset', 'rb') as f:
    all_input  = np.load(f, allow_pickle = True)
    all_output = np.load(f, allow_pickle = True)

tmp = np.random.randint(0, all_input.shape[0], 100)
all_input  = all_input[tmp]
all_output = all_output[tmp]

all_input_mean = np.mean(all_input, axis=0)
all_input_std  = np.std(all_input, axis=0)

all_output_mean = np.mean(all_output, axis=0)
all_output_std  = np.std (all_output, axis=0)

#all_input  = np.random.uniform(-0.3, 0.3, (50000, 6))
#all_output = np.sum(all_input, 1, keepdims = True)

count = all_output.shape[0]

print(all_input.shape)
print(all_output.shape)

print(np.count_nonzero(np.isnan(all_input)))
print(np.count_nonzero(np.isnan(all_output)))

# Debug
print(f"Before normalization - Input:  mean = {np.mean(all_input,  axis=0)}, std = {np.std(all_input,  axis=0)}")
print(f"Before normalization - Output: mean = {np.mean(all_output, axis=0)}, std = {np.std(all_output, axis=0)}")

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
fig.savefig('fig.png')

# Normalize
all_input  = (all_input  - all_input_mean) / all_input_std
#all_input  = (all_input  - np.amin(all_input,  axis=0)) / (np.amax(all_input,  axis=0) - np.amin(all_input,  axis=0)) * 2.0 - 1.0
#all_input = (all_input - np.median(all_input, axis=0)) / np.std(all_input, axis=0)
#all_input = (all_input - np.median(all_input, axis=0)) / (np.amax(all_input,  axis=0) - np.amin(all_input,  axis=0))

all_output = (all_output - all_output_mean) / all_output_std
#all_output = (all_output - np.amin(all_output, axis=0)) / (np.amax(all_output, axis=0) - np.amin(all_output, axis=0)) * 2.0 - 1.0
#all_output = (all_output - np.median(all_output, axis=0)) / np.std(all_output, axis=0)
#all_output = (all_output - np.median(all_output, axis=0)) / (np.amax(all_output, axis=0) - np.amin(all_output, axis=0))

# Debug
print(f"After normalization - Input:  mean = {np.mean(all_input,  axis=0)}, std = {np.std(all_input,  axis=0)}")
print(f"After normalization - Output: mean = {np.mean(all_output, axis=0)}, std = {np.std(all_output, axis=0)}")

sub_count = int(count / 10)
for i in range(10):
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

    model = Model()

    print("Testing on testing data")
    test(model, testing_input, testing_output)
    print("Testing on training data")
    test(model, training_input, training_output)

    while True:
        train(model, 1, 1, 0.05, training_input, training_output)
        print("During training - Testing on testing data")
        test(model, testing_input, testing_output)
        print("During training - Testing on training data")
        test(model, training_input, training_output)

    print("Testing on testing data")
    test(model, testing_input, testing_output)
    print("Testing on training data")
    test(model, training_input, training_output)
    sys.exit(1)

