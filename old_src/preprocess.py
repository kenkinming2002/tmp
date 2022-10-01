#!/usr/bin/env python

from merra import *
from aod   import *

from pathlib import Path

import gc
import os
import math

import numpy as np
import scipy.io as sio

from netCDF4 import Dataset

from tqdm import tqdm

# Load data from matlab matrix file

class PM25:
    def __init__(self):
        self.LA   = get_data_ml(f"data/PM2.5/LA_PM25.mat")
        self.LO   = get_data_ml(f"data/PM2.5/LO_PM25.mat")
        self.PM25 = get_data_ml(f"data/PM2.5/PM25.mat")

    def get(self, time, station):
        return (self.LA[0][station], self.LO[0][station], self.PM25[time][station])


def create_dataset():
    data_aod = AOD()

    data_pblh  = MERRA("data/Meteorological_Data/PBLH", "PBLH")
    data_ps    = MERRA("data/Meteorological_Data/TWPR", "PS")
    data_qv10m = MERRA("data/Meteorological_Data/TWPR", "QV10M") # Probably relative humidity
    data_t2m   = MERRA("data/Meteorological_Data/TWPR", "T2M")
    data_u10m  = MERRA("data/Meteorological_Data/TWPR", "U10M")
    data_v10m  = MERRA("data/Meteorological_Data/TWPR", "V10M")

    data_pm25 = PM25()

    input_list  = list()
    output_list = list()
    for time in tqdm(range(59 * 24), desc="Matching"):
        for station in range(1653):
            la, lo, pm25 = data_pm25.get(time, station)
            if math.isnan(pm25):
                continue

            aod = data_aod.get(la, lo, time)

            pblh  = data_pblh.get(la, lo, time)
            ps    = data_ps.get(la, lo, time)
            qv10m = data_qv10m.get(la, lo, time)
            t2m   = data_t2m.get(la, lo, time)
            u10m  = data_u10m.get(la, lo, time)
            v10m  = data_v10m.get(la, lo, time)


            if aod is None or pblh is None or ps is None or qv10m is None or t2m is None or u10m is None or v10m is None:
                continue

            input_list.append(np.array([aod, pblh, ps, qv10m, t2m, math.sqrt(u10m ** 2 + v10m ** 2)]))
            output_list.append(np.array([pm25]))

    all_input  = np.stack(input_list)
    all_output = np.stack(output_list)

    with open('dataset', 'wb') as f:
        np.save(f, all_input)
        np.save(f, all_output)

create_dataset()

