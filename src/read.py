#!/usr/bin/env python

import sys
import numpy as np

with open('dataset', 'rb') as f:
    all_input  = np.load(f, allow_pickle = True)
    all_output = np.load(f, allow_pickle = True)

all_input = all_input * np.array([1, 1/2000, 1/200000, 100, 1/500, 1/10])
all_output /= 100

print(all_input.shape)
print(all_output.shape)

index = int(sys.argv[1])

print(all_input[index])
print(all_output[index])

