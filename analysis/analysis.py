import numpy as np
import json
import os
import h5py
import re

def get_experiments(filename):
    f = open(filename, 'r')
    lines = '\n'.join(f.readlines())
    lines = "[" + lines[:-1].rstrip(',') + "]"
    return json.loads(lines)

# data is a list of dictionaries, all of which must have the same keys. 
def make_stacked_barchart(ax, x_positions, width, data, keys, tick_labels, vertical=True):
    num_bars = len(data)
    #colors = ['green', 'goldenrod', 'slateblue', 'pink', 'brown']

    bar_edges = np.zeros(num_bars)
    for i in range(len(keys)):
        bar_data = np.array([data[j][keys[i]] for j in range(num_bars)]) 

        if vertical:
            ax.bar(x_positions, bar_data, width, bottom=bar_edges, edgecolor='black', label=keys[i])
        else:
            ax.barh(x_positions, bar_data, width, left=bar_edges, edgecolor='black', label=keys[i])

        bar_edges += bar_data

    if vertical:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(tick_labels)
    else:
        ax.set_yticks(x_positions)
        ax.set_yticklabels(tick_labels)

    ax.grid(True)

def load_factors_from_file(filename):
    factors = {}
    with h5py.File(filename, 'r') as f:
        keys = f.keys()

        for key in keys:
            if 'FACTOR_MODE' in key: 
                factors[key] = f[key][:]

    return factors

def read_splatt_trace(filename):
    '''
    Returns a list of times taken for each iteration of SPLATT
    '''
    p = re.compile('.*\((.+s)\).*')
    
    iteration_times = []

    with open(filename, 'r') as f:
        lines = f.readlines()

        for line in lines:
            m = p.match(line)

            if m is not None:
                iteration_times.append(float(m.group(1)[:-1]))

    return iteration_times

def dummy():
    ax,fig=plt.subplots()

    bar_width = 0.35
    ngroups=1
    index = np.arange(ngroups)
    plt.bar(index, [normalized_time_exact_amazon], bar_width, color='g', label='Exact ALS (Ours)')
    plt.bar(index + bar_width, [normalized_time_splatt_amazon], bar_width, color='y', label='SPLATT')
    plt.bar(index + 2 * bar_width, [total_time_sketched_amazon], bar_width, color='purple', label='Lev. Scores, s=131k')

    plt.xticks(index + bar_width, ['Amazon, 15 Iter.'])
    plt.ylabel("Runtime (s)")

    plt.xlim(-1,2)
    #plt.set_xticklabels( ('Amazon, 15 Iter.'))

    plt.legend()