#!/usr/bin/env python

from base import *

aod = np.load(f"stage3/AOD.npy")

pblh  = np.load(f"stage3/PBLH.npy")
ps    = np.load(f"stage3/PS.npy")
qv10m = np.load(f"stage3/QV10M.npy")
t2m   = np.load(f"stage3/T2M.npy")
u10m  = np.load(f"stage3/U10M.npy")
v10m  = np.load(f"stage3/V10M.npy")

pm25  = np.load(f"stage3/PM25.npy")

ws = np.sqrt(np.square(u10m) + np.square(v10m))

input_data = np.stack((aod, pblh, ps, qv10m, t2m, ws), axis = -1)
output_data = np.expand_dims(pm25, axis = -1)

print(input_data.shape)
print(output_data.shape)

os.makedirs(f"stage4", exist_ok = True)

np.save("stage4/input.npy",  input_data)
np.save("stage4/output.npy", output_data)
