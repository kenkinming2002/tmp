import numpy as np
import matplotlib.pyplot as plt

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

