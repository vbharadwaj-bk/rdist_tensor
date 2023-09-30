#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import numpy.linalg as la
import matplotlib
import matplotlib.pyplot as plt
import json, re, sys, os
import matplotlib.gridspec as gridspec

sys.path.append("..")
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 11})

colors = {'sts_cp': 'green', 'splatt': 'royalblue', 'cp_arls_lev': 'darkorange'}


# In[3]:


def parse_splatt_trace(filepath):
    with open(filepath, 'r') as f:
        text = f.read()

        # Split the text into segments using the ***** delimiter
        segments = re.split(r'\*{64,}', text)
        segments = [segment for segment in segments if 'splatt v2.0.0' in segment]
        
        # Define the regular expression pattern to extract fit vs. time data
        #pattern = r'its = (\d+) \((\d+\.\d+s)\)  fit = ([\d.]+)'
        pattern = r'its\s*=\s*(\d+)\s*\(([\d.]+s)\)\s*fit\s*=\s*([\d.]+)'

        # Initialize a list to store time/fit/iteration pairs for each segment
        segment_fit_data = []

        # Process each segment separately
        for segment in segments:
             # Extract fit and time values for this segment
            segment_data = {'iterations': [], 'times': [], 'fits': []}
            factor_pattern = r'NFACTORS=(\d+)'

            # Use re.search to find the match
            factor_match = re.search(factor_pattern, segment)

            # Check if a match was found
            if factor_match:
                # Extract the NFACTORS value from the match
                segment_data["rank"] = int(factor_match.group(1))
            
            # Use regular expressions to extract fit vs. time data from each segment
            matches = re.findall(pattern, segment)

            for match in matches:
                iteration, time, fit = match
                segment_data['iterations'].append(int(iteration))
                segment_data['times'].append(float(time[:-1]))
                segment_data['fits'].append(float(fit))

            segment_data['times'] = np.cumsum(segment_data['times'])
            # Append the data for this segment to the list
            segment_fit_data.append(segment_data)
            
        return segment_fit_data
        
def generate_trace_ours(directory, tensor_name, algorithm, sample_count, rank):
    exps = []
    for filepath in os.listdir(directory):
        if filepath.endswith('.out'):
            with open(os.path.join(directory, filepath), 'r') as f:
                exps.append(json.load(f))
            
    filtered_exps = []
    for exp in exps:
        if exp['input'] == tensor_name and exp['algorithm'] == algorithm \
            and exp['sample_count'] == sample_count and exp['target_rank'] == rank:
            filtered_exps.append(exp)
            
    data = []
    
    for exp in filtered_exps:
        exp_data = {}
        exp_data['iterations'] = exp['stats']['rounds']
        exp_data['times'] = exp['stats']['als_times']
        exp_data['fits'] = exp['stats']['fits']
        
        data.append(exp_data)
        
    return data

def generate_interpolation(data):
    '''
    Data must be structured as a list of dictionaries, each with keys 'iterations', 'times', and 'fits'
    that each point to a list.
    '''
    max_time = np.max([np.max(trace['times']) for trace in data])
    x_axis = np.linspace(0, max_time, 10000)
        
    interpolations = []
    for trace in data:
        max_fits = [np.max(trace['fits'][:i]) for i in range(1,len(trace['fits']) + 1)]
        interp_y = np.interp(x_axis, xp=trace['times'], fp=max_fits)
        interpolations.append(interp_y)
            
    interpolations = np.array(interpolations)
    mean_interp = np.mean(interpolations, axis=0)
        
    return x_axis, mean_interp


# In[36]:


fig, ax = plt.subplots()

x_splatt, y_splatt = generate_interpolation(parse_splatt_trace('../data/fit_progress_vs_time/reddit_trace_splatt.txt'))
x_sts_cp, y_sts_cp = generate_interpolation(generate_trace_ours('../data/fit_progress_vs_time', 'reddit', 'sts_cp', 98304, 100))
x_cp_arls_lev, y_cp_arls_lev = generate_interpolation(generate_trace_ours('../data/fit_progress_vs_time', 'reddit', 'cp_arls_lev', 98304, 100))

ax.plot(x_splatt, y_splatt, label="SPLATT", c=colors['splatt'])
ax.plot(x_sts_cp, y_sts_cp, label="d-STS-CP (ours)",c=colors['sts_cp'])
ax.plot(x_cp_arls_lev, y_cp_arls_lev, label="d-CP-ARLS-LEV (ours)",c=colors['cp_arls_lev'])
ax.scatter(x_splatt[-1], y_splatt[-1], marker='*',c=colors['splatt'])
ax.scatter(x_sts_cp[-1], y_sts_cp[-1], marker='*',c=colors['sts_cp'])
ax.scatter(x_cp_arls_lev[-1], y_cp_arls_lev[-1], marker='*',c=colors['cp_arls_lev'])

ax.axhline(max(y_splatt), linestyle='--', c=colors['splatt'])

ax.grid(True)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Fit")
ax.set_ylim([0.07, 0.11])
ax.legend(loc='lower right', framealpha=1.0)
fig.savefig('figures/fit_vs_time_reddit.pdf')


# In[79]:


# Information for the highlight slug

print(max(x_splatt))
print(max(y_splatt))
print(max(x_sts_cp))
print(max(y_sts_cp))
print(1494.1029/159.578)
print(max(x_cp_arls_lev))
print(max(y_cp_arls_lev))
x_splatt[np.argwhere(y_splatt > max(y_cp_arls_lev))[0]]


# In[27]:


# Generate the accuracy table for our algorithm
tensor_print_map = {
    "uber": "Uber",
    "amazon": "Amazon",
    "reddit": "Reddit"
}
result_map = {}

table = ""
for tensor in ["uber", "amazon", "reddit"]:
    first = True
    for rank in [25, 50, 75]:
        for alg in ['cp_arls_lev', 'sts_cp']:
            traces = generate_trace_ours('../data/accuracy_benchmarks', tensor, alg, 65536, rank)
            if len(traces) < 5:
                print(f"{tensor} {rank} {alg} incomplete! Trace count is {len(traces)}")
            max_fits = [np.max(trace['fits']) for trace in traces]
            mean_fit = np.mean(max_fits)
            std_fit = np.std(max_fits)
            result_map[(tensor, rank, alg)] = (mean_fit, std_fit)
            
        if first:
            first = False
            table += "\\multirow{3}{*}{" + tensor_print_map[tensor] + "}   \n"
        
        if tensor != "reddit":
            table += f"& {rank}   & {result_map[tensor, rank, 'cp_arls_lev'][0]:.3f} & {result_map[tensor, rank, 'sts_cp'][0]:.3f} & 0 \\\\ \n"
        else:
             table += f"& {rank}   & {result_map[tensor, rank, 'cp_arls_lev'][0]:.4f} & {result_map[tensor, rank, 'sts_cp'][0]:.4f} & 0 \\\\ \n"
    table += "\\midrule\n"
    
table += "\\bottomrule\n"
    
print(table)


# In[37]:


# data is a list of dictionaries, all of which must have the same keys. 
def make_stacked_barchart(ax, x_positions, width, data, keys, tick_labels=None, vertical=True, label=True):
    num_bars = len(data)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
 '#7f7f7f', '#bcbd22', '#17becf'] # Default Matplotlib color cycle

    bar_edges = np.zeros(num_bars)
    for i in range(len(keys)):
        bar_data = np.array([data[j][keys[i]] for j in range(num_bars)]) 
        
        if label:
            label_to_use = keys[i]
        else:
            label_to_use = None
            
        if vertical:
            func = ax.bar
        else:
            func = ax.barh
        
        func(x_positions, bar_data, width, bottom=bar_edges, edgecolor='black', label=label_to_use, color=colors[i])
        bar_edges += bar_data

    if tick_labels is not None:
        if vertical:
            ax.set_xticks(x_positions)
            ax.set_xticklabels(tick_labels)
        else:
            ax.set_yticks(x_positions)
            ax.set_yticklabels(tick_labels)

    ax.grid(True)


# In[34]:


# Strong scaling experiments

def get_strong_scaling_measurements(directory, fields):
    exps = []
    for filepath in os.listdir(directory):
        if filepath.endswith('.out'):
            with open(os.path.join(directory, filepath), 'r') as f:
                exps.append(json.load(f))
            
    filtered_exps = []
    for exp in exps:
        match = True
        for key in fields:
            if fields[key] != exp[key]:
                match = False
        
        if match:
            filtered_exps.append(exp)
            
    results = {}
    
    stat_fields = ['leverage_sampling_time', 'row_gather_time', 'design_matrix_prep_time', 'spmm_time', 
              'dense_reduce_time', 'postprocessing_time', 'sampler_update_time']
    
    for field in stat_fields:
        if len(filtered_exps) > 0:
            results[field] = np.mean([exp['stats'][field]['mean'] for exp in filtered_exps])
        else:
            results[field] = 0.0
        
    if len(filtered_exps) > 0:
        results['als_time']=np.mean([exp['stats']['als_total_time'] for exp in filtered_exps])
    else:
        results[field] = 0.0
            
    return results

def stats_to_bar(stats):
    bar = { 'Allgather': stats['row_gather_time'],
            'Reduce-Scatter': stats['dense_reduce_time'],
            'Sampling': stats['leverage_sampling_time'],
            'MTTKRP': stats['design_matrix_prep_time'] + stats['spmm_time'],
            'Postprocessing': stats['sampler_update_time'] + stats['postprocessing_time']
          }
    return bar

num_tensors=3

fig = plt.figure(tight_layout=True)
fig.set_size_inches(8 * (1.1), 8 * (1.1))
spec = fig.add_gridspec(2, num_tensors, left=0.0, right=0.8)
axs = [[fig.add_subplot(spec[i, j]) for j in range(num_tensors)] for i in range(2)]

#for i in range(1, num_tensors):
#    axs[i].sharey(axs[0])
#    axs[i].label_outer()

for j in range(2):
    for i in range(num_tensors):
        axs[j][i].grid(True)
        
        if j == 1:
            axs[j][i].set_xlabel("Node Count")

            
axs[0][0].set_ylabel("d-CP-ARLS-LEV")
axs[1][0].set_ylabel("d-STS-CP")


tensors = ['amazon', 'patents', 'reddit']
algs = ['cp_arls_lev', 'sts_cp']
tensor_node_counts = {'amazon': [1, 2, 4, 8], 'patents': [2, 4, 8, 16], 'reddit': [2, 4, 8, 16]}

for idx1, tensor in enumerate(tensors):
    for idx2, alg in enumerate(algs):
        exps = []
        node_counts = tensor_node_counts[tensor]
        for node_count in node_counts:
            exps.append(get_strong_scaling_measurements('../data/strong_scaling', 
                                                        {
                                                        'input': tensor,
                                                        'algorithm': alg,
                                                        'sample_count': 65536,
                                                        'target_rank': 25,
                                                        'node_count': str(node_count)
                                                        }))
            
        x = np.log(node_counts) / np.log(2)
        bars = [stats_to_bar(exp) for exp in exps]
        make_stacked_barchart(axs[idx2][idx1], x, 0.25, bars, list(bars[0].keys()), node_counts, vertical=True, label=(idx1+idx2==0))
        axs[idx2][idx1].set_axisbelow(True)
    axs[0][idx1].set_title(tensor.capitalize())

fig.text(-0.1, 0.5, 'Time for 20 ALS Iterations (s)', va='center', rotation='vertical')
fig.legend(bbox_to_anchor=(0.83,0.05),ncol=5)
fig.savefig("figures/strong_scaling_25_cp_arls_lev.pdf", bbox_inches='tight')
fig.show()


# In[15]:


# Generate comparisons to the baseline
tensors = {"amazon":
        {"splatt_filename": "../data/baseline_runtime_comparison/amazon_baseline_4.txt",
         "sample_count": 65536},
           "patents":
        {"splatt_filename": "../data/baseline_runtime_comparison/patents_baseline_4.txt",
         "sample_count": 65536},
        "reddit":
        {"splatt_filename": "../data/baseline_runtime_comparison/reddit_baseline_4.txt",
         "sample_count": 65536}
       }

output = {}

ranks = [25, 50, 75]

for tensor in tensors.keys():
    splatt_traces = parse_splatt_trace(tensors[tensor]["splatt_filename"])
    
    for rank in ranks:
        trace = [trace for trace in splatt_traces if trace["rank"] == rank][0]
        avg_time_per_iteration = trace['times'][-1] / trace['iterations'][-1]
        output[(tensor, rank, 'splatt')] = avg_time_per_iteration
        
        for alg in ['cp_arls_lev', 'sts_cp']:
            trace =  generate_trace_ours('../data/baseline_runtime_comparison', tensor, alg, tensors[tensor]["sample_count"], rank)[0]
            avg_time_per_iteration = trace['times'][-1] / trace['iterations'][-1]
            output[(tensor, rank, alg)] = avg_time_per_iteration

            
num_tensors = 3
fig = plt.figure(tight_layout=True)
fig.set_size_inches(6 * (1.1), 4 * (1.1))
spec = fig.add_gridspec(1, num_tensors, left=0.0, right=0.8)
axs = [[fig.add_subplot(spec[i, j]) for j in range(num_tensors)] for i in range(1)]

for i in range(1):
    for j in range(num_tensors):
        axs[i][j].sharey(axs[0][0])
        axs[i][j].grid(True)
        if i + j != 0:
            axs[i][j].label_outer()
            
print_names = {'splatt': 'SPLATT', 'sts_cp': 'd-STS-CP', 'cp_arls_lev': 'd-CP-ARLS-LEV'}
labelled = False
for idx, tensor in enumerate(tensors.keys()):
    axs[0][idx].axhline(1.0, linestyle='--', c=colors['splatt'])
    axs[0][idx].set_title(tensor.capitalize())
    axs[0][idx].set_xticks([25, 50, 75])
    axs[0][idx].set_axisbelow(True)
    for idx_alg, alg in enumerate(['splatt', 'sts_cp', 'cp_arls_lev']):
        width=5
        heights = np.array([output[tensor, rank, 'splatt'] for rank in ranks]) / np.array([output[tensor, rank, alg] for rank in ranks])
        print(heights)
        
        if not labelled:
            label = print_names[alg]
        else:
            label = None
        
        axs[0][idx].bar(np.array(ranks) + width * (idx_alg - 1), heights, width=5, label=label, edgecolor='black', color=colors[alg])
        axs[0][idx].tick_params(axis='x', which='major',length=0)
        
    if not labelled:
        labelled = True
        
fig.text(-0.1, 0.5, 'Speedup over SPLATT', va='center', rotation='vertical')
fig.text(0.3, 0.03, 'Target Rank', va='center')
fig.legend(bbox_to_anchor=(0.72,-0.02),ncol=3)
fig.savefig('figures/speedup_over_splatt.pdf',  bbox_inches='tight')


# In[61]:


fig, ax = plt.subplots()
ax.grid(True)

tensors = ['amazon', 'patents', 'reddit']
dists = ['tensor_stationary', 'accumulator_stationary']

all_x = []
labels = []
mids = []
for idx1, tensor in enumerate(tensors):
    exps = []
    for idx2, dist in enumerate(dists):
        exps.append(get_strong_scaling_measurements('../data/communication_comparison', 
                                                        {
                                                        'input': tensor,
                                                        'algorithm': 'sts_cp',
                                                        'sample_count': 65536,
                                                        'target_rank': 25,
                                                        'data_distribution': dist
                                                        }))
    mid = idx1 * 3.5
    mids.append(mid)
    x = [mid-0.25,mid+0.25]
    all_x += x
    labels += ['TS', 'AS']
    bars = [stats_to_bar(exp) for exp in exps]
    make_stacked_barchart(ax, x, 0.5, bars, list(bars[0].keys()), vertical=True, label=(idx1==0))
    
for idx, mid in enumerate(mids):
    ax.text(mid, -3.0, tensors[idx].capitalize(), ha='center')
    
ax.set_axisbelow(True)
ax.set_xticks(all_x)
ax.set_xticklabels(labels)

ax.set_ylabel('Time for 20 ALS Iterations (s)')
fig.legend(bbox_to_anchor=(0.79,0.00),ncol=2)
fig.savefig("figures/communication_schedule_comparison.pdf", bbox_inches='tight')
fig.show()


# In[123]:


def get_raw_trace(directory, tensor_name, algorithm, sample_count, rank):
    exps = []
    for filepath in os.listdir(directory):
        if filepath.endswith('.out'):
            with open(os.path.join(directory, filepath), 'r') as f:
                exps.append(json.load(f))
            
    filtered_exps = []
    for exp in exps:
        if exp['input'] == tensor_name and exp['algorithm'] == algorithm \
            and exp['sample_count'] == sample_count and exp['target_rank'] == rank:
            filtered_exps.append(exp)
            
    return filtered_exps

step = 16
fig, ax = plt.subplots(figsize=(7,5))

tensors = ['amazon', 'patents', 'reddit']
tensor_ranks = {'amazon': [16* i for i in range(2,9)], 'patents': [16* i for i in range(2,9)], 'reddit': [16* i for i in range(2,9)]}

for tensor in tensors:
    avg_throughputs = []
    for rank in tensor_ranks[tensor]:
        exps = get_raw_trace('../data/weak_scaling', tensor, 'sts_cp', 65536, rank)
        throughput = np.mean([exp["stats"]["sum_nonzeros_iterated"] / exp["stats"]["als_times"][-1]  for exp in exps])
        print(f'{tensor}, {throughput}')
        avg_throughputs.append(throughput)
        
    ax.plot(np.array(tensor_ranks[tensor]) / step, avg_throughputs, '-o', label=tensor.capitalize())
    
ax.set_ylim(0,1e9)
ax.set_xlabel("Node Count | Rank")
ax.set_ylabel("Throughput (NNZ / s)")
ax.set_xticks([i for i in range(2, 9)])
ax.set_xticklabels([f'{i} | {step * i}' for i in range(2, 9)])
    
ax.grid(True)
ax.legend()
fig.savefig("figures/weak_scaling.pdf")


# In[124]:


# In[ ]:





# In[ ]:




