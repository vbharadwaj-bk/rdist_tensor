import numpy as np
import json
import os

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
