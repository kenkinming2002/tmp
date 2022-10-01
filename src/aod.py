from base import *

def get_data_ml(filename):
    f = sio.loadmat(filename)
    name = Path(filename).stem
    data = f[name]
    return data

# Return list of scatter data indexed by time
# Scatter data of the form (aod, lat, lon)
def load_aod():
    def rounded_TIME_as_time(rounded_TIME):
        date = int(rounded_TIME[4:7])-1
        hour = int(rounded_TIME[8:10])
        return date * 24 + hour

    aod_list  = []
    lat_list  = []
    lon_list  = []
    time_list = []

    for month in [1, 2]:
        for station in ["MOD", "MYD"]:
            AOD_LA_LO    = get_data_ml(f"data/AOD/{month}/{station}/AOD_LA_LO.mat")
            rounded_TIME = get_data_ml(f"data/AOD/{month}/{station}/rounded_TIME.mat")

            aod  = AOD_LA_LO[0]
            lat  = AOD_LA_LO[1]
            lon  = AOD_LA_LO[2]

            time = np.vectorize(rounded_TIME_as_time)(rounded_TIME)

            aod_list.append(aod)
            lat_list.append(lat)
            lon_list.append(lon)
            time_list.append(time)

    aod  = np.concatenate(aod_list)
    lat  = np.concatenate(lat_list)
    lon  = np.concatenate(lon_list)
    time = np.concatenate(time_list)

    res = []
    for i in range(24 * 59):
        mask = time == i
        res.append((aod[mask], lat[mask], lon[mask]))

    return res

