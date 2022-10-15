#!/usr/bin/env python

from base import *

input_data  = np.load("stage4/input.npy")
output_data = np.load("stage4/output.npy")

aod   = input_data[...,0]
pblh  = input_data[...,1]
ps    = input_data[...,2]
qv10m = input_data[...,3]
t2m   = input_data[...,4]
ws    = input_data[...,5]

pm25 = output_data[...,0]

def plot_histogram(name, data):
    fig, ax = plt.subplots(1, 1)
    ax.hist(data, bins = 500)
    ax.set_title(name)
    fig.savefig(f"output/histogram/{name}.png")

os.makedirs(f"output/histogram", exist_ok = True)

plot_histogram("AOD",   aod)
plot_histogram("PBLH",  pblh)
plot_histogram("PS",    ps)
plot_histogram("QV10M", qv10m)
plot_histogram("T2M",   t2m)
plot_histogram("WS",    ws)
plot_histogram("PM25",  pm25)

def plot_scatter(name1, name2, data1, data2):
    fig, ax = plt.subplots(1, 1)
    ax.scatter(data1, data2)
    ax.set_xlabel(name1)
    ax.set_ylabel(name2)
    fig.savefig(f"output/scatter/{name1}-{name2}.png")

os.makedirs(f"output/scatter", exist_ok = True)

plot_scatter("AOD",   "PM25", aod  , pm25)
plot_scatter("PBLH",  "PM25", pblh , pm25)
plot_scatter("PS",    "PM25", ps   , pm25)
plot_scatter("QV10M", "PM25", qv10m, pm25)
plot_scatter("T2M",   "PM25", t2m  , pm25)
plot_scatter("WS",    "PM25", ws   , pm25)


