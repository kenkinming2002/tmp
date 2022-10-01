from base import *

# Return list of scatter data indexed by time
# Grid data indexed by [lat][lon]
def get_data_nc(filename, name):
    f = Dataset(filename)
    data = f[name][:]
    assert(not np.ma.is_masked(data))
    return data.filled()

def load_merra(dirname, name):
    def filename_as_date(filename):
        month = int(filename[31:33])
        day   = int(filename[33:35])
        if month == 1:
            return day - 1
        elif month == 2:
            return day + 31 - 1
        else:
            raise Exception(f"Invalid month={month}")

    res = [None] * 24 * 59
    for filename in tqdm(os.listdir(dirname), desc=f"MERRA(dirname={dirname},name={name})"):
        date = filename_as_date(filename)
        data = get_data_nc(f"{dirname}/{filename}", name)
        for hour in range(11):
            time = date * 24 + hour
            res[time] = data[hour]

    return res
