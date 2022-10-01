#!/usr/bin/env python

from aod   import *
from merra import *
from pm25  import *

data_all_aod   = load_aod()
data_all_pblh  = load_merra("data/Meteorological_Data/PBLH", "PBLH")
data_all_ps    = load_merra("data/Meteorological_Data/TWPR", "PS")
data_all_qv10m = load_merra("data/Meteorological_Data/TWPR", "QV10M") # Probably relative humidity
data_all_t2m   = load_merra("data/Meteorological_Data/TWPR", "T2M")
data_all_u10m  = load_merra("data/Meteorological_Data/TWPR", "U10M")
data_all_v10m  = load_merra("data/Meteorological_Data/TWPR", "V10M")
data_all_pm25  = load_pm25()

# Match over every time
for time in range(24 * 59):
    data_aod   = data_all_aod[time]

    data_pblh  = data_all_pblh[time]
    data_ps    = data_all_ps[time]
    data_qv10m = data_all_qv10m[time]
    data_t2m   = data_all_t2m[time]
    data_u10m  = data_all_u10m[time]
    data_v10m  = data_all_v10m[time]

    data_pm25  = data_all_pm25[time]
