#!/usr/bin/env python

import sys
import numpy as np

with open('dataset', 'rb') as f:
    all_input  = np.load(f, allow_pickle = True)
    all_output = np.load(f, allow_pickle = True)

print(all_input.shape)
print(all_output.shape)

index = int(sys.argv[1])

print(all_input[index])
print(all_output[index])

