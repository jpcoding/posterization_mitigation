import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# get rid of type3 fonts 
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
datastes = [  'miranda', 'hurricane' ,'nyx' ,'jhtdb_pressure'] 
data_szie = [ 384*384*256, 500*500*100, 512**3, 512**3 ]
results = {}
for dataset in datastes:
    results[dataset] = (np.load("time_array_{}.npy".format(dataset)))

datastes = [  'miranda', 'hurricane' ,'nyx' ,'jhtdb_pressure'] 
dataset_titles = ['Miranda', 'Hurricane' ,'NYX' ,'JHTDB-Small'] 
figs,axs = plt.subplots(2, 2, figsize=(6, 6), constrained_layout=True)
cores_ticks = [1, 2, 4, 8, 16, 32,64] 
# cores = np.arange(1, len(cores_ticks)+1)
cores = np.array([1, 2, 4, 8, 16, 32, 64])
ebs = [1e-4, 1e-3, 1e-2]
EB_strs = ['1E-4', '1E-3', '1E-2'] 
colors = ['tab:blue', 'tab:orange', 'tab:green']
markers = ['o', 's', '^'] 
line_styles = ['-', '--', ':'] 
for i, dataset in enumerate(datastes):
    cur_time = np.mean(np.mean(results[dataset],axis= 1),axis=1)
    cur_speed = data_szie[i]*4.0/1024**2 / cur_time 
    for j in range(3):
        axs[i//2, i%2].plot(cores, cur_speed[:,j], label=f"$\epsilon=$"+f"{EB_strs[j]}",
                            marker=markers[j], linestyle=line_styles[j], color=colors[j],linewidth=2)
    axs[i//2, i%2].legend(loc='upper left', fontsize=12, frameon=True, 
                           ) 
    axs[i//2, i%2].set_title(dataset_titles[i], fontsize=16)
    axs[i//2, i%2].set_xlabel("Number of Threads",fontsize=14)
    axs[i//2, i%2].set_xscale('log')
    axs[i//2, i%2].set_xticks(cores_ticks)
    axs[i//2, i%2].set_xticklabels(cores_ticks, size=12)
    axs[i//2, i%2].tick_params(axis='y', labelsize=12)
    if(i%2==0) : axs[i//2, i%2].set_ylabel("Throughput (MB/s)", fontsize=14)
    axs[i//2, i%2].grid()
figs.savefig("omp_results.pdf", bbox_inches='tight')