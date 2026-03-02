import os 
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib.ticker import ScalarFormatter

def read_cr_info(file):
    filenames=[]
    crs=[] 
    with open(file, 'r') as f: 
        lines = f.readlines()
        filenames = eval(lines[0])
        crs = eval(lines[1]) 
    # cobvert to dataframe 
    
    np_crs = np.zeros((len(filenames), len(crs[0])))
    for i in range(len(crs)):
        np_crs[i] = crs[i] 
    
    headers = [0.0001, 0.0005, 0.001, 0.005, 0.01 ]
    df = pd.DataFrame(np_crs, columns=headers, index=filenames)
    return df 


mkdir = './figs'  
if not os.path.exists(mkdir):
    os.makedirs(mkdir)

datasets = ['miranda', 'hurricane', 'nyx', 'jhtdb_pressure']
results = {}
for dataset in datasets:
    results[dataset] = {}
    results[dataset]['cusz_cr']= (read_cr_info(f'../cusz_compression/{dataset}_cusz.log'))
    results[dataset]['cuszp_cr']= (read_cr_info(f'../cuszp_compression/{dataset}_cuszp.log'))
    results[dataset]['orig_psnr'] =  pd.read_csv(f'psnr_{dataset}_orig.csv', index_col=0)  
    results[dataset]['orig_ssim'] =  pd.read_csv(f'ssim_{dataset}_orig.csv', index_col=0)  
    results[dataset]['idw_psnr'] = pd.read_csv(f'psnr_{dataset}_post_3d.csv', index_col=0) 
    results[dataset]['idw_ssim'] = pd.read_csv(f'ssim_{dataset}_post_3d.csv', index_col=0) 
    results[dataset]['cusz_agg_cr'] = len(results[dataset]['cusz_cr'].to_numpy())/np.sum(1/results[dataset]['cusz_cr'].to_numpy(),axis=0)
    results[dataset]['cuszp_agg_cr'] = len(results[dataset]['cuszp_cr'].to_numpy())/np.sum(1/results[dataset]['cuszp_cr'].to_numpy(),axis=0)
    results[dataset]['orig_agg_psnr'] = results[dataset]['orig_psnr'].loc['agg_psnr']
    results[dataset]['orig_agg_ssim'] = results[dataset]['orig_ssim'].loc['agg_ssim']
    results[dataset]['idw_agg_ssim'] = results[dataset]['idw_ssim'].loc['agg_ssim']
    results[dataset]['idw_agg_psnr'] = results[dataset]['idw_psnr'].loc['agg_psnr']
    results[dataset]['gaussian_ssim'] = pd.read_csv(f'gaussian_{dataset}post_ssim.csv', index_col=0).loc['agg_ssim']
    results[dataset]['gaussian_psnr'] = pd.read_csv(f'gaussian_{dataset}post_psnr.csv', index_col=0).loc['agg_psnr']


## plot ssim figures 
figs, axs = plt.subplots(4,3, figsize=(8, 12), constrained_layout=True)    
datastes_names = ['Miranda', 'Hurricane', 'NYX', 'JHTDB-Small'] 
line_styles = ['-', '--', '-.', ':']
line_colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange']
markers = ['o', 's', '^', 'D']
# plot cusz cr in the first row 
ebs = [0.0001, 0.0005, 0.001, 0.005, 0.01 ]
# ebs = [1,2, 3 ,4 ,5 ]
eb_labels = ['1E-4', '5E-4', '1E-3', '5E-3', '1E-2'] 
line_styles = ['-', '--', '-.', ':']
line_colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange']
markers = ['o', 's', '^', 'D']
labels = ['Quantized', 'cuSZ', 'cuSZp']
for i, dataset in enumerate(datasets): 
    xs  = [ebs, 32.0/results[dataset]['cusz_agg_cr'], 32.0/results[dataset]['cuszp_agg_cr']]
    ys = [results[dataset]['orig_agg_ssim'], results[dataset]['idw_agg_ssim'], results[dataset]['gaussian_ssim']]
    for j in range(3):
        axs[i, j].plot(xs[j], ys[0], label=labels[j], linestyle=line_styles[0], color=line_colors[0], marker=markers[0])
        axs[i, j].plot(xs[j], ys[2], label='Gaussian', linestyle=line_styles[3], color=line_colors[3], marker=markers[3])
        axs[i, j].plot(xs[j], ys[1], label='Ours', linestyle=line_styles[1], color=line_colors[1], marker=markers[1])
        if(j==0):
                axs[i,j].invert_xaxis()
                axs[i,j].set_xlabel('Relative Error', fontsize=14) 
                axs[i, j].set_xticks(ebs)
                axs[i, j].set_xticklabels(eb_labels)
                axs[i, j].set_xscale('log')
        else:
                axs[i,j].set_xlabel('Bit-rate', fontsize=14)
        if(j==0 ):
            axs[i,j].set_ylabel('SSIM', fontsize=14)
        axs[i,j].legend(frameon=True, framealpha=0.3, loc='lower right', fontsize=9)
        axs[i,j].grid()
        axs[i,j].tick_params(axis='both', which='major', labelsize=12)
axs[0,0].set_title('EB-SSIM')  
axs[0,1].set_title('cuSZ Rate-SSIM')
axs[0,2].set_title('cuSZp Rate-SSIM')

figs.savefig('./figs/ssim_cr_vertical.pdf',bbox_inches='tight')


## plot psnr figures 
figs, axs = plt.subplots(4,3, figsize=(8, 12), constrained_layout=True)    
datastes_names = ['Miranda', 'Hurricane', 'NYX', 'JHTDB-Small'] 
line_styles = ['-', '--', '-.', ':']
line_colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange']
markers = ['o', 's', '^', 'D']
# plot cusz cr in the first row 
ebs = [0.0001, 0.0005, 0.001, 0.005, 0.01 ]
eb_labels = ['1E-4', '5E-4', '1E-3', '5E-3', '1E-2'] 
line_styles = ['-', '--', '-.', ':']
line_colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange']
markers = ['o', 's', '^', 'D']
labels = ['Quantized', 'cuSZ', 'cuSZp']
for i, dataset in enumerate(datasets): 
    xs  = [ebs, 32.0/results[dataset]['cusz_agg_cr'], 32.0/results[dataset]['cuszp_agg_cr']]
    ys = [results[dataset]['orig_agg_psnr'], results[dataset]['idw_agg_psnr'],results[dataset]['gaussian_psnr']]
    for j in range(3):
        axs[i, j].plot(xs[j], ys[0], label=labels[j], linestyle=line_styles[0], color=line_colors[0], marker=markers[0])
        axs[i, j].plot(xs[j], ys[2], label='Gaussian', linestyle=line_styles[3], color=line_colors[3], marker=markers[3])
        axs[i, j].plot(xs[j], ys[1], label='Ours', linestyle=line_styles[1], color=line_colors[1], marker=markers[1])
        if(j==0):
                axs[i,j].invert_xaxis()
                axs[i,j].set_xlabel('Relative Error', fontsize=14) 
                axs[i, j].set_xticks(ebs)
                axs[i, j].set_xticklabels(eb_labels)
                axs[i, j].set_xscale('log')
        else:
                axs[i,j].set_xlabel('Bit-rate', fontsize=14)
        if(j==0 ):
            axs[i,j].set_ylabel('PSNR', fontsize=14)
        axs[i,j].legend(frameon=True, framealpha=0.3, loc='best', fontsize=9)
        axs[i,j].grid()
        axs[i,j].tick_params(axis='both', which='major', labelsize=12)
axs[0,0].set_title('EB-PSNR')  
axs[0,1].set_title('cuSZ Rate-PSNR')
axs[0,2].set_title('cuSZp Rate-PSNR')

figs.savefig('./figs/psnr_cr_vertical.pdf',bbox_inches='tight')
