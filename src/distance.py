import numpy as np
import math

def distance_haversine(la1, lo1, la2, lo2):
    ratio = math.pi / 180
    tmp1 =                                             np.square(np.sin((la1 - la2) * ratio / 2.0))
    tmp2 = np.cos(la1 * ratio) * np.cos(la2 * ratio) * np.square(np.sin((lo1 - lo2) * ratio / 2.0))
    return np.arcsin(np.sqrt(np.clip(tmp1 + tmp2, 0.0, 1.0)))
