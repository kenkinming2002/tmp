import numpy as np
import matplotlib.pyplot as plt

##### Load #####
with open('dataset', 'rb') as f:
    all_input  = np.load(f)
    all_output = np.load(f)

#all_input  = np.random.uniform(-0.2, 0.2, (50000, 6))
#all_output = np.sum(all_input, 1, keepdims = True)

##### Shuffling #####
def shuffle(input_data, output_data):
    assert(input_data.shape[0] == output_data.shape[0])
    tmp = np.random.permutation(input_data.shape[0])
    input_data  = input_data[tmp]
    output_data = output_data[tmp]
    return (input_data, output_data)

all_input, all_output = shuffle(all_input, all_output)

##### Normalization #####

### Unchanged
#def normalize_input(data):
#    return data
#
#def normalize_output(data):
#    return data
#
#def denormalize_input(data):
#    return data
#
#def denormalize_output(data):
#    return data

### Standardize
#input_mean  = np.mean(all_input, axis=0)
#input_std   = np.std (all_input, axis=0)
#
#output_mean = np.mean(all_output, axis=0)
#output_std  = np.std (all_output, axis=0)
#
#def normalize_input(data):
#    return (data - input_mean) / input_std
#
#def normalize_output(data):
#    return (data - output_mean) / output_std
#
#def denormalize_input(data):
#    return data * input_std + input_mean
#
#def denormalize_output(data):
#    return data * output_std + output_mean

### Min/max
input_min = np.amin(all_input, axis=0)
input_max = np.amax(all_input, axis=0)

output_min = np.amin(all_output, axis=0)
output_max = np.amax(all_output, axis=0)

def normalize_input(data):
    return (data - input_min) / (input_max - input_min) * 2.0 - 1.0

def normalize_output(data):
    return (data - output_min) / (output_max - output_min) * 2.0 - 1.0

def denormalize_input(data):
    return (data + 1.0) / 2.0 * (input_max - input_min) + input_min

def denormalize_output(data):
    return (data + 1.0) / 2.0 * (output_max - output_min) + output_min

##### Selection #####
#tmp = np.random.randint(0, all_input.shape[0], 10000)
#all_input  = all_input[tmp]
#all_output = all_output[tmp]

##### Debugging #####
print(all_input.shape)
print(all_output.shape)

print(np.count_nonzero(np.isnan(all_input)))
print(np.count_nonzero(np.isnan(all_output)))

