#!/usr/bin/env python

from pathlib import Path

import numpy as np
import scipy.io as sio

from netCDF4 import Dataset

import sys

# Load data from matlab matrix file
def get_data_ml(filename):
    return data

# Load data from NetCDF file
def get_data_nc(filename, name):
    return data

def usage():
    print('usage: show.py filename [name]')
    sys.exit(-1)

def arg_get(i):
    if len(sys.argv) <= i:
        usage()
    return sys.argv[i]

filename = arg_get(1)
extension = Path(filename).suffix
if extension == '.mat':
    f = sio.loadmat(filename)
    name = Path(filename).stem
    data = f[name]
    print(data.dtype)
    print(data.shape)
    print(data)
elif extension == '.nc':
    f = Dataset(filename)
    print(f)
    for name in f.variables:
        print(f[name])
        data = f[name][:]
        print(name)
        print(data.dtype)
        print(data.shape)
        print(data)
else:
    usage()

