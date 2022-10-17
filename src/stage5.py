#!/usr/bin/env python

from base import *

def put_data_ml(filename, data):
    name = Path(filename).stem
    sio.savemat(filename, {name : data})

input_data  = np.load("stage4/input.npy")
output_data = np.load("stage4/output.npy")

os.makedirs(f"stage5", exist_ok = True)

put_data_ml("stage5/input.mat", input_data)
put_data_ml("stage5/output.mat", output_data)
