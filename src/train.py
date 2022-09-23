#!/usr/bin/env python

from tqdm import tqdm

import sys
import math
import numpy as np

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
        return 1.0 / (1.0 + np.exp(-self.saved_input))

    def backward(self, output_gradient):
        tmp = np.exp(-self.saved_input)
        return output_gradient / ((1+tmp)*(1+1/tmp))

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

    def train(self, epoch, batch_size, learning_rate, training_input, training_output):
        size        = len(training_input)
        batch_count = int(size / batch_size)
        for k in range(epoch):
            for i in tqdm(range(batch_count), desc=f"Training(epoch={k})"):
                batch_input  = training_input[i*batch_size:(i+1)*batch_size]
                batch_output = training_output[i*batch_size:(i+1)*batch_size]

                value = batch_input

                value = self.layer1.forward(value)
                value = self.layer2.forward(value)
                value = self.layer3.forward(value)
                value = self.layer4.forward(value)

                gradient = (value - batch_output) * 2

                gradient = self.layer4.backward(gradient)
                gradient = self.layer3.backward(gradient)
                gradient = self.layer2.backward(gradient)
                gradient = self.layer1.backward(gradient)

                self.layer1.train(learning_rate)
                self.layer2.train(learning_rate)
                self.layer3.train(learning_rate)
                self.layer4.train(learning_rate)

    def test(self, testing_input, testing_output):
        value = testing_input

        value = self.layer1.forward(value)
        value = self.layer2.forward(value)
        value = self.layer3.forward(value)
        value = self.layer4.forward(value)

        for i in range(5):
            print(f"Example: prediction = {value[i]}, output={testing_output[i]}")

        ss_res = np.square(testing_output - value).sum()
        ss_tot = np.square(testing_output - testing_output.mean()).sum()
        r2 = 1 - ss_res / ss_tot
        print(f"Testing:ss_res={ss_res}, ss_tot={ss_tot}, R2={r2}")

# Load
with open('dataset', 'rb') as f:
    all_input  = np.load(f, allow_pickle = True)
    all_output = np.load(f, allow_pickle = True)

print(all_input.shape)
print(all_output.shape)

all_input  = np.random.uniform(0, 0.1, (10000, 6))
all_output = np.sum(all_input, 1, keepdims = True)

count = all_output.shape[0]

# Shuffle
tmp = np.random.permutation(count)
all_input  = all_input[tmp]
all_output = all_output[tmp]

# Normalize
all_input  /= np.amax(all_input, axis=0)
all_output /= np.amax(all_output)

sub_count = int(count / 10)
for i in range(10):
    training_input  = np.concatenate((all_input [:i*sub_count], all_input [(i+1)* sub_count:]))
    training_output = np.concatenate((all_output[:i*sub_count], all_output[(i+1)* sub_count:]))

    testing_input  = all_input [i * sub_count: (i+1)*sub_count]
    testing_output = all_output[i * sub_count: (i+1)*sub_count]
    model = Model()

    print("Testing on testing data")
    model.test(testing_input, testing_output)
    print("Testing on training data")
    model.test(training_input, training_output)

    model.train(20, 64, 0.05, training_input, training_output)

    print("Testing on testing data")
    model.test(testing_input, testing_output)
    print("Testing on training data")
    model.test(training_input, training_output)

    sys.exit(0)

