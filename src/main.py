#!/usr/bin/env python

from aod   import *
from merra import *
from pm25  import *

aod_data = load_aod()
for i in range(10):
    print(aod_data[i])

merra_data = load_merra("data/Meteorological_Data/PBLH", "PBLH")
for i in range(24 * 59):
    if merra_data[i] is None:
        print("None")
    else:
        print(merra_data[i].shape)

pm25_data = load_pm25()
for i in range(10):
    print(list(pm25_data[i][0]))
