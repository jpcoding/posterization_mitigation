import numpy as np 
import pandas as pd 
import re
import glob

def get_time(file):
    time = []
    count = 0
    start_reading = False  # Flag to track when to start reading
    with open(file, 'r') as f:
        for line in f:
            if not start_reading:
                # Check for the first occurrence of "Running with rel_eb=0.0001"
                if "Running with rel_eb=0.001" in line:
                    start_reading = True
            elif re.search('Rank 0, time:', line):
                time.append(float(line.split()[-1]))
                count += 1
                if count == 5:
                    break
    print(time)
    time = np.sort(time)[1:4].mean()
    # time = np.mean(time) 
    # print(time)
    time = np.asarray(time).mean()
    return time

prefix = ['em', 'full', 'opt']
configs = ['8x8x8', '8x8x4', '8x4x4', '4x4x4'][::-1]
ext = '.out'
dir = './plot_logs/'
file_lists = []
for p in prefix:
    for c in configs:
        pattern = f'{dir}{p}_{c}*{ext}'
        matched_files = glob.glob(pattern)
        file_lists.extend(matched_files)
print(len(file_lists))
        
time = np.zeros((len(prefix) * len(configs)))
for i, file in enumerate(file_lists):
    print(file) 
    time[i] = get_time(file)
    
    # time[i] = time_
    # print(time_) 
# for i, file in enumerate(file_lists):
    
time = time.reshape(len(prefix), len(configs))
fils_size = 4096**3*4.0/(1024**2)
speed = fils_size/time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# get rid of type3 fonts 
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
labesls = ["Embarssingly parallel", "Exact Parallelization", "Approximate Parallelization"]
cores = [1, 2, 3, 4]
markers = ['o', 's', '^'] 

cores_tags  = [64, 128, 256, 512]
fig, ax = plt.subplots(figsize=(4,4))
for i in range(len(prefix)):
    ax.plot(cores, speed[i]/1024, marker=markers[i], label=labesls[i],linewidth=4.0,markersize=10) 
ax.set_xticks(cores)
ax.set_ylim([0, 6.0])
ax.set_xticklabels(cores_tags)
ax.set_xlabel('Number of cores', fontsize=20)
ax.set_ylabel('Throughput (GB/s)', fontsize=20)
plt.title('Strong scaling', fontsize=20) 
plt.grid()
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
fig.savefig('strong_scaling.pdf', bbox_inches='tight')

# save legend as a separate file plot it as one line 
fig_leg = plt.figure(figsize=(6, 0.5))  # Adjust width as needed
ax_leg = fig_leg.add_subplot(111)
# Create the legend handles and labels from the original plot
handles, labels = ax.get_legend_handles_labels()
leg = ax_leg.legend(handles, labels, loc='center', ncol=len(labels), frameon=True,fontsize=20,
                    handlelength=1.0, handleheight=0.5, columnspacing=0.5, labelspacing=0.005)  # Adjust legend spacing
                
# legend spcing adjustment to 0.1 
# Hide axes
ax_leg.axis('off')
# Save the legend figure
fig_leg.savefig("strong_scaling_legend.pdf", bbox_inches='tight')