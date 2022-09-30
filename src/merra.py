from base import *
from tqdm import tqdm

import os
import math

MERRA_THRESHOLD = 0.005

MERRA_LA_BEGIN      = -90.0
MERRA_LA_END        = 90.0
MERRA_LA_RESOLUTION = 0.5

MERRA_LO_BEGIN      = -180.0
MERRA_LO_END        = 179.375
MERRA_LO_RESOLUTION = 0.625

MERRA_HOUR_BEGIN      = 0.5
MERRA_HOUR_END        = 10.5
MERRA_HOUR_RESOLUTION = 1

#                                [59][11][361][576]
# Store data in array indexed by [date][hour][la][lo]
class MERRA:
    def __init__(self, dirname, name):
        self.data = [None] * 59
        for filename in tqdm(os.listdir(dirname), desc=f"MERRA(dirname={dirname},name={name})"):
            month = int(filename[31:33])
            day   = int(filename[33:35])
            date = day if month == 1 else 31 + day
            self.data[date-1] = get_data_nc(f"{dirname}/{filename}", name)

    def get(self, la, lo, time):
        date        = int(time / 24)
        hour        = time % 24

        date_index = date
        la_index   = round((la - MERRA_LA_BEGIN) / MERRA_LA_RESOLUTION)
        lo_index   = round((lo - MERRA_LO_BEGIN) / MERRA_LO_RESOLUTION)
        hour_index = round((hour - MERRA_HOUR_BEGIN) / MERRA_HOUR_RESOLUTION) # What if hour == 0

        if hour_index >= 11 or hour_index < 0:
            return None

        the_la = MERRA_LA_BEGIN + la_index * MERRA_LA_RESOLUTION
        the_lo = MERRA_LO_BEGIN + lo_index * MERRA_LO_RESOLUTION

        ratio = math.pi / 180
        tmp1 =                                                   math.sin((the_la - la) * ratio / 2.0) ** 2
        tmp2 = math.cos(la * ratio) * math.cos(the_la * ratio) * math.sin((the_lo - lo) * ratio / 2.0) ** 2
        distance = math.asin(math.sqrt(max(0.0, min(tmp1 + tmp2, 1.0))))

        if distance > MERRA_THRESHOLD:
            return None

        return self.data[date_index][hour_index][la_index][lo_index]

