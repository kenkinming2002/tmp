#!/usr/bin/env python

from layer import *

import sys
import math

input_data  = np.load("stage4/input.npy")
output_data = np.load("stage4/output.npy")

#input_data  = np.random.uniform(-0.2, 0.2, (50000, 6))
#output_data = np.sum(input_data, 1, keepdims = True)

def shuffle(input_data, output_data):
    assert(input_data.shape[0] == output_data.shape[0])
    tmp = np.random.permutation(input_data.shape[0])
    input_data  = input_data[tmp]
    output_data = output_data[tmp]
    return (input_data, output_data)

input_data, output_data = shuffle(input_data, output_data)

### Min/Max
#input_min = np.amin(input_data, axis=0)
#input_max = np.amax(input_data, axis=0)
#
#output_min = np.amin(output_data, axis=0)
#output_max = np.amax(output_data, axis=0)
#
#def normalize_input(data):
#    return (data - input_min) / (input_max - input_min) * 2.0 - 1.0
#
#def normalize_output(data):
#    return (data - output_min) / (output_max - output_min) * 2.0 - 1.0
#
#def denormalize_input(data):
#    return (data + 1.0) / 2.0 * (input_max - input_min) + input_min
#
#def denormalize_output(data):
#    return (data + 1.0) / 2.0 * (output_max - output_min) + output_min

### Standardize
input_mean  = np.mean(input_data, axis=0)
input_std   = np.std (input_data, axis=0)

output_mean = np.mean(output_data, axis=0)
output_std  = np.std (output_data, axis=0)

def normalize_input(data):
    return (data - input_mean) / input_std

def normalize_output(data):
    return (data - output_mean) / output_std

def denormalize_input(data):
    return data * input_std + input_mean

def denormalize_output(data):
    return data * output_std + output_mean

def train(model, epoch, batch_size, learning_rate, momentum, training_input, training_output):
    training_input  = normalize_input(training_input)
    training_output = normalize_output(training_output)

    size        = len(training_input)
    batch_count = int(size / batch_size)
    for k in range(epoch):
        for i in tqdm(range(batch_count), desc=f"Training(epoch={k})"):
            batch_input  = training_input[i*batch_size:(i+1)*batch_size]
            batch_output = training_output[i*batch_size:(i+1)*batch_size]

            batch_prediction = model.forward(batch_input)

            batch_output_gradient = (batch_prediction - batch_output) * 2
            batch_input_gradient  = model.backward(batch_output_gradient)

            model.train(learning_rate, momentum)

def test(title, model, testing_input, testing_output):
    testing_input  = normalize_input(testing_input)
    testing_output = normalize_output(testing_output)

    prediction = model.forward(testing_input)

    testing_input  = denormalize_input(testing_input)
    testing_output = denormalize_output(testing_output)
    prediction     = denormalize_output(prediction)

    ss_res = np.square(testing_output - prediction).sum()
    ss_tot = np.square(testing_output - testing_output.mean()).sum()
    r2 = 1 - ss_res / ss_tot
    print(f"Testing({title} dataset):ss_res={ss_res}, ss_tot={ss_tot}, R2={r2}")

    ##### Plotting #####
    fig, ax = plt.subplots(1, 1)
    ax.scatter(testing_output, prediction)
    ax.set_xlabel("Output")
    ax.set_ylabel("Prediction")
    plt.show()

count = input_data.shape[0]
sub_count = int(count / 10)
for i in range(10):
    training_input  = np.concatenate((input_data [:i*sub_count], input_data [(i+1)* sub_count:]))
    training_output = np.concatenate((output_data[:i*sub_count], output_data[(i+1)* sub_count:]))

    testing_input  = input_data [i * sub_count: (i+1)*sub_count]
    testing_output = output_data[i * sub_count: (i+1)*sub_count]

    model = Model()
    test("testing",  model, testing_input,  testing_output)
    test("training", model, training_input, training_output)

    #testing_input,  testing_output  = shuffle(testing_input,  testing_output)
    #training_input, training_output = shuffle(training_input, training_output)
    train(model, 10, 64, 0.01, 0.9, training_input, training_output)

    test("testing",  model, testing_input,  testing_output)
    test("training", model, training_input, training_output)

