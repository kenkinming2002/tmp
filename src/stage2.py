#!/usr/bin/env python

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import sys

from match import *

def match_at(time, pbar):
    try:
        # Loading
        aod_data = np.load(f"stage1/{time:0>4}/AOD/data.npy")
        aod_la   = np.load(f"stage1/{time:0>4}/AOD/la.npy")
        aod_lo   = np.load(f"stage1/{time:0>4}/AOD/lo.npy")

        pblh_data  = np.load(f"stage1/{time:0>4}/PBLH/data.npy")
        ps_data    = np.load(f"stage1/{time:0>4}/PS/data.npy")
        qv10m_data = np.load(f"stage1/{time:0>4}/QV10M/data.npy")
        t2m_data   = np.load(f"stage1/{time:0>4}/T2M/data.npy")
        u10m_data  = np.load(f"stage1/{time:0>4}/U10M/data.npy")
        v10m_data  = np.load(f"stage1/{time:0>4}/V10M/data.npy")

        pm25_data = np.load(f"stage1/{time:0>4}/PM25/data.npy")
        pm25_la   = np.load(f"stage1/{time:0>4}/PM25/la.npy")
        pm25_lo   = np.load(f"stage1/{time:0>4}/PM25/lo.npy")

        # Matching
        aod_indices, aod_distances, aod_match_la1, aod_match_lo1, aod_match_la2, aod_match_lo2 = match_scatter_scatter(pm25_la, pm25_lo, aod_la, aod_lo)
        merra_la_indices, merra_lo_indices = match_scatter_grid(pm25_la, pm25_lo, -90.0, 0.5, -180.0, 0.625)

        aod_mask = aod_distances <= 5

        aod = aod_data[aod_indices][aod_mask]

        aod_match_la1 = aod_match_la1[aod_mask]
        aod_match_lo1 = aod_match_lo1[aod_mask]
        aod_match_la2 = aod_match_la2[aod_mask]
        aod_match_lo2 = aod_match_lo2[aod_mask]

        pblh  = pblh_data [(merra_la_indices, merra_lo_indices)][aod_mask]
        ps    = ps_data   [(merra_la_indices, merra_lo_indices)][aod_mask]
        qv10m = qv10m_data[(merra_la_indices, merra_lo_indices)][aod_mask]
        t2m   = t2m_data  [(merra_la_indices, merra_lo_indices)][aod_mask]
        u10m  = u10m_data [(merra_la_indices, merra_lo_indices)][aod_mask]
        v10m  = v10m_data [(merra_la_indices, merra_lo_indices)][aod_mask]

        pm25 = pm25_data[aod_mask]

        # Saving
        os.makedirs(f"stage2/{time:0>4}", exist_ok = True)

        np.save(f"stage2/{time:0>4}/AOD.npy", aod)

        np.save(f"stage2/{time:0>4}/PBLH.npy",  pblh)
        np.save(f"stage2/{time:0>4}/PS.npy",    ps)
        np.save(f"stage2/{time:0>4}/QV10M.npy", qv10m)
        np.save(f"stage2/{time:0>4}/T2M.npy",   t2m)
        np.save(f"stage2/{time:0>4}/U10M.npy",  u10m)
        np.save(f"stage2/{time:0>4}/V10M.npy",  v10m)

        np.save(f"stage2/{time:0>4}/PM25.npy",  pm25)
    except FileNotFoundError:
        pass

    pbar.update(1)

with tqdm(total=24*59, desc="Matching", position=0, leave=False) as pbar:
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = [ex.submit(match_at, i, pbar) for i in range(24*59)]
        for future in as_completed(futures):
            result = future.result()

