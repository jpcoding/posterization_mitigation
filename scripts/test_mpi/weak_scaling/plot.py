
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# get rid of type3 fonts 
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

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
                if "Running with " in line:
                    start_reading = True
            elif re.search('Rank 0, time:', line):
                time.append(float(line.split()[-1]))
                count += 1
                if count ==5 :
                    break
    # print(file)
    # print(time)
    # time = np.asarray(time).mean()
    time = np.sort(time)[1:4].mean()
    # time = np.mean(time)
    return time
prefix = ['em', 'full', 'opt']
configs = ['8x8x8', '8x8x4', '8x4x4', '4x4x4'][::-1]
mpi_dims = [[8,8,8],[8,8,4],[8,4,4],[4,4,4]][::-1]

block_size = 512 
mpi_dims = np.asarray(mpi_dims) * block_size
file_size = np.prod(mpi_dims,axis=1) * 4.0 / 1024 / 1024
# print(file_size)

orig_dims = [] 
ext = '.out'
dir = './'
file_lists = []
for p in prefix:
    for c in configs:
        pattern = f'{dir}{p}_{c}*{ext}'
        matched_files = glob.glob(pattern)
        file_lists.extend(matched_files)
# print(len(file_lists))    
time = np.zeros((len(prefix) * len(configs)))
for i, file in enumerate(file_lists):
    time[i] = get_time(file)
time = time.reshape(len(prefix), len(configs))

speed = np.zeros((len(prefix), len(configs)))
for i in range(len(prefix)):
    speed[i] = file_size / time[i]
# print(speed)
labesls = ["Embarssingly parallel", "Exact parallel", "Approximate parallel"]

markers = ['o', 's', '^'] 

cores = [1, 2, 3, 4]
cores_tags  = [64, 128, 256, 512]

fig, ax = plt.subplots(figsize=(4,4))
for i in range(len(prefix)):
    ax.plot(cores, speed[i]/1024.0, marker=markers[i], label=labesls[i],linewidth=2.0, markersize=8)
#set x axis to log scale
ax.set_ylim([0, 6.0])
ax.set_xticks(cores)
ax.set_xticklabels(cores_tags)
# plt.legend()
plt.title('Weak scaling ',fontsize=22) 
plt.xlabel('Number of cores',fontsize=20)
plt.ylabel('Throughput (GB/s)',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid()
fig.savefig('weak_scaling.pdf', bbox_inches='tight')