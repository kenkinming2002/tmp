import numpy as np
import matplotlib.pyplot as plt

##### Load #####
with open('dataset', 'rb') as f:
    all_input  = np.load(f)
    all_output = np.load(f)

#all_input  = np.random.uniform(-0.3, 0.3, (50000, 6))
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
input_mean  = np.mean(all_input, axis=0)
input_std   = np.std (all_input, axis=0)

output_mean = np.mean(all_output, axis=0)
output_std  = np.std (all_output, axis=0)

def normalize_input(data):
    return (data - input_mean) / input_std

def normalize_output(data):
    return (data - output_mean) / output_std

def denormalize_input(data):
    return data * input_std + input_mean

def denormalize_output(data):
    return data * output_std + output_mean

##### Selection #####
#tmp = np.random.randint(0, all_input.shape[0], 10000)
#all_input  = all_input[tmp]
#all_output = all_output[tmp]

##### Plotting #####
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

##### Debugging #####
count = all_output.shape[0]

print(all_input.shape)
print(all_output.shape)

print(np.count_nonzero(np.isnan(all_input)))
print(np.count_nonzero(np.isnan(all_output)))

