from base import *
from tqdm import tqdm

def deg2rad(value):
    return value * (math.pi / 180)

def match_scatter_grid(la, lo, la_begin, la_step, lo_begin, lo_step):
    la_indices = np.rint((la - la_begin) / la_step).astype(int)
    lo_indices = np.rint((lo - lo_begin) / lo_step).astype(int)
    return (la_indices, lo_indices)

def match_scatter_scatter(la1, lo1, la2, lo2):
    la1 = deg2rad(la1)
    lo1 = deg2rad(lo1)
    la2 = deg2rad(la2)
    lo2 = deg2rad(lo2)

    assert(la1.size == lo1.size)
    count = la1.size

    indices   = []
    distances = []
    for i in tqdm(range(count), desc="scatter scatter match", position=1, leave=False):
        the_la1 = la1[i]
        the_lo1 = lo1[i]
        the_la2 = la2
        the_lo2 = lo2

        tmp1 = np.square(np.sin((the_la1 - the_la2) / 2.0))
        tmp2 = np.square(np.sin((the_lo1 - the_lo2) / 2.0)) * np.cos(the_la1) * np.cos(the_la2)

        sub_distances =  2 * 6371 * np.arcsin(np.sqrt(np.clip(tmp1 + tmp2, 0.0, 1.0)))

        index    = np.argmin(sub_distances)
        distance = sub_distances[index]

        indices.append(index)
        distances.append(distance)

    return (np.array(indices), np.array(distances))

