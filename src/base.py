from pathlib import Path
from netCDF4 import Dataset

import numpy as np
import scipy.io as sio

def get_data_ml(filename):
    f = sio.loadmat(filename)
    name = Path(filename).stem
    data = f[name]
    return data

# Load data from NetCDF file
def get_data_nc(filename, name):
    f = Dataset(filename)
    data = f[name][:]
    return data

