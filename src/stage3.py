#!/usr/bin/env python

from base import *

aod_list   = []

pblh_list  = []
ps_list    = []
qv10m_list = []
t2m_list   = []
u10m_list  = []
v10m_list  = []

pm25_list  = []

# Load
for time in range(24*59):
    try:
        aod   = np.load(f"stage2/{time:0>4}/AOD.npy")

        pblh  = np.load(f"stage2/{time:0>4}/PBLH.npy")
        ps    = np.load(f"stage2/{time:0>4}/PS.npy")
        qv10m = np.load(f"stage2/{time:0>4}/QV10M.npy")
        t2m   = np.load(f"stage2/{time:0>4}/T2M.npy")
        u10m  = np.load(f"stage2/{time:0>4}/U10M.npy")
        v10m  = np.load(f"stage2/{time:0>4}/V10M.npy")

        pm25  = np.load(f"stage2/{time:0>4}/PM25.npy")

        aod_list.append(aod)

        pblh_list.append(pblh)
        ps_list.append(ps)
        qv10m_list.append(qv10m)
        t2m_list.append(t2m)
        u10m_list.append(u10m)
        v10m_list.append(v10m)

        pm25_list.append(pm25)
    except FileNotFoundError:
        pass

# Concatenate
aod = np.concatenate(aod_list)

pblh = np.concatenate(pblh_list)
ps = np.concatenate(ps_list)
qv10m = np.concatenate(qv10m_list)
t2m = np.concatenate(t2m_list)
u10m = np.concatenate(u10m_list)
v10m = np.concatenate(v10m_list)

pm25 = np.concatenate(pm25_list)

# Save
os.makedirs(f"stage3", exist_ok = True)

np.save(f"stage3/AOD.npy", aod)

np.save(f"stage3/PBLH.npy",  pblh)
np.save(f"stage3/PS.npy",    ps)
np.save(f"stage3/QV10M.npy", qv10m)
np.save(f"stage3/T2M.npy",   t2m)
np.save(f"stage3/U10M.npy",  u10m)
np.save(f"stage3/V10M.npy",  v10m)

np.save(f"stage3/PM25.npy",  pm25)
