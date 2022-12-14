from base import *
from distance import *

from tqdm import tqdm

import math

AOD_THRESHOLD = 0.005

class AOD:
    def __init__(self):
        def parse_time(time_str):
            date = int(time_str[4:7])-1
            hour = int(time_str[8:10])
            return date * 24 + hour

        aod_list  = [[] for i in range(59 * 24)]
        la_list   = [[] for i in range(59 * 24)]
        lo_list   = [[] for i in range(59 * 24)]

        for month in [1, 2]:
            for station in ["MOD", "MYD"]:
                AOD_LA_LO    = get_data_ml(f"data/AOD/{month}/{station}/AOD_LA_LO.mat")
                rounded_TIME = get_data_ml(f"data/AOD/{month}/{station}/rounded_TIME.mat")

                aod  = AOD_LA_LO[0]
                la   = AOD_LA_LO[1]
                lo   = AOD_LA_LO[2]
                time = np.vectorize(parse_time)(rounded_TIME)

                # Preprocess and store them on a grid indexed by time
                for the_aod, the_la, the_lo, the_time in tqdm(zip(aod, la, lo, time), desc=f"AOD(month={month},station={station})"):
                    aod_list[the_time].append(the_aod)
                    la_list[the_time].append(the_la)
                    lo_list[the_time].append(the_lo)

        self.aod_data = [np.array(l) for l in aod_list]
        self.la_data  = [np.array(l) for l in la_list]
        self.lo_data  = [np.array(l) for l in lo_list]

    # Find aod based on la lo and time
    def get(self, la, lo, time):
        distances = distance_haversine(la, lo, self.la_data[time], self.lo_data[time])
        distances = np.ma.array(distances, mask = distances > AOD_THRESHOLD)
        if distances.mask.all():
            return None

        i = np.ma.argmin(distances)

        aod_data = self.aod_data[time]
        return aod_data[i]
