#!/usr/bin/env python

from load import *

AOD   = all_input[...,0]
PBLH  = all_input[...,1]
PS    = all_input[...,2]
QV10M = all_input[...,3]
T2M   = all_input[...,4]
WS    = all_input[...,5]

PM25 = all_output[...,0]

def plot_histogram(name, data):
    fig, ax = plt.subplots(1, 1)
    ax.hist(data, bins = 500)
    ax.set_title(name)
    fig.savefig(f"output/histogram/{name}.png")

plot_histogram("AOD",   AOD)
plot_histogram("PBLH",  PBLH)
plot_histogram("PS",    PS)
plot_histogram("QV10M", QV10M)
plot_histogram("T2M",   T2M)
plot_histogram("WS",    WS)
plot_histogram("PM25",  PM25)

def plot_scatter(name1, name2, data1, data2):
    fig, ax = plt.subplots(1, 1)
    ax.scatter(data1, data2)
    ax.set_xlabel(name1)
    ax.set_ylabel(name2)
    fig.savefig(f"output/scatter/{name1}-{name2}.png")

plot_scatter("AOD",   "PM25", AOD  , PM25)
plot_scatter("PBLH",  "PM25", PBLH , PM25)
plot_scatter("PS",    "PM25", PS   , PM25)
plot_scatter("QV10M", "PM25", QV10M, PM25)
plot_scatter("T2M",   "PM25", T2M  , PM25)
plot_scatter("WS",    "PM25", WS   , PM25)

