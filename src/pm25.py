from base import *

def get_data_ml(filename):
    f = sio.loadmat(filename)
    name = Path(filename).stem
    data = f[name]
    return data

# Return list of scatter data indexed by time
# Scatter data of the form (pm25, lat, lon)
def load_pm25():
    LA   = get_data_ml(f"data/PM2.5/LA_PM25.mat")
    LO   = get_data_ml(f"data/PM2.5/LO_PM25.mat")
    PM25 = get_data_ml(f"data/PM2.5/PM25.mat")

    res = []
    for i in range(24 * 59):
        pm25 = PM25[i]
        la   = LA[0]
        lo   = LO[0]

        # Mask invalid data
        mask = ~np.isnan(pm25)
        res.append((pm25[mask], la[mask], lo[mask]))

    return res
