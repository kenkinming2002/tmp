#!/usr/bin/env python

from aod   import *
from merra import *
from pm25  import *

def write_scatter(name, data_all):
    for time in range(24 * 59):
        if data_all[time][0].size == 0:
            continue

        os.makedirs(f"stage1/{time:0>4}/{name}", exist_ok = True)
        np.save    (f"stage1/{time:0>4}/{name}/data.npy", data_all[time][0])
        np.save    (f"stage1/{time:0>4}/{name}/la.npy",   data_all[time][1])
        np.save    (f"stage1/{time:0>4}/{name}/lo.npy",   data_all[time][2])


def write_grid(name, data_all):
    for time in range(24 * 59):
        if data_all[time] is None:
            continue

        os.makedirs(f"stage1/{time:0>4}/{name}", exist_ok = True)
        np.save    (f"stage1/{time:0>4}/{name}/data.npy", data_all[time])

write_scatter("AOD", load_aod())

write_grid("PBLH",  load_merra("data/Meteorological_Data/PBLH", "PBLH" ))
write_grid("PS",    load_merra("data/Meteorological_Data/TWPR", "PS"   ))
write_grid("QV10M", load_merra("data/Meteorological_Data/TWPR", "QV10M"))
write_grid("T2M",   load_merra("data/Meteorological_Data/TWPR", "T2M"  ))
write_grid("U10M",  load_merra("data/Meteorological_Data/TWPR", "U10M" ))
write_grid("V10M",  load_merra("data/Meteorological_Data/TWPR", "V10M" ))

write_scatter("PM25", load_pm25())

