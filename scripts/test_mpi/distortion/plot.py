import numpy as np 
import matplotlib.pyplot as plt



psnr = np.loadtxt('psnr.txt').reshape(5,2)
ssim = np.loadtxt('ssim.txt').reshape(5,2)
ebs = [0.0001, 0.0005 , 0.001, 0.005 ,0.01]

data = [psnr[:,0], ssim[:,0]][::-1]
data_opt = [psnr[:,1], ssim[:,1]][::-1]
method = ['Quantized data', 'Approximate parallelization'] 
figs, axs = plt.subplots(1, 2, figsize=(7, 4)) 
data_tag = ['PSNR', 'SSIM'][::-1]
for j in range(2):
    axs[j].plot(ebs, data[j], label=method[0],marker='o',color='tab:blue')   
    axs[j].plot(ebs, data_opt[j], label=method[1],
                marker='D',color='tab:red',linestyle='-',linewidth=2) 
    axs[j].set_xlabel('Relative Error Bound', fontsize=14)
    axs[j].set_ylabel(data_tag[j], fontsize=14)
    # axs[j].set_title(data_tag[j])
    axs[j].set_xscale('log')
    axs[j].invert_xaxis()
    axs[j].grid()
    axs[j].tick_params(axis='x', labelsize=14)
    axs[j].tick_params(axis='y', labelsize=14)



# put legend outside and let them share it 
handles, labels = axs[0].get_legend_handles_labels()

# Place shared legend outside the plots (below)
figs.legend(handles, labels, loc='lower center', ncol=len(labels),
            bbox_to_anchor=(0.55, -0.04), frameon=True,
            fontsize=14, handlelength=1.5, handletextpad=0.5  )

# Save the figure with extra space for the legend
figs.tight_layout()
figs.subplots_adjust(bottom=0.25)  #

# axs[0].set_ylim(0, 50)
figs.savefig('jhtdb-large_rate-distortion.pdf', dpi=300, bbox_inches='tight')