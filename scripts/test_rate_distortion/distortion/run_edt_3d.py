import os 
import sys
import time
import numpy as np
import matplotlib.pyplot as plt 
import ctypes
import subprocess
import pandas as pd
import re 
import toml 

# with open('../configs.toml', 'r') as f:
#     config = toml.load(f)

exe_file_path = "../../../install/bin/test/test_quantize_and_edt"



def qcatssim(orig:np.ndarray[np.float32], decompressed:np.ndarray[np.float32]):
    dims = np.array(orig.shape)[::-1]
    # print(dims)
    lib = ctypes.CDLL("../../../install/lib/liblibqcatssim.so")
    lib.calculateSSIM.restype = ctypes.c_double
    lib.calculateSSIM.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32),
                              np.ctypeslib.ndpointer(dtype=np.float32),
                              np.ctypeslib.ndpointer(dtype=np.int32),
                              ctypes.c_int]
    result =lib.calculateSSIM(
        orig, 
        decompressed,
        dims.astype(np.int32), 
        ctypes.c_int(dims.size))
    return result 



def verify( src_data, dec_data):
    """
    Compare the decompressed data with original data
    :param src_data: original data, numpy array
    :param dec_data: decompressed data, numpy array
    :return: max_diff, psnr, nrmse
    """
    data_range = np.max(src_data) - np.min(src_data)
    diff = src_data - dec_data
    max_diff = np.max(abs(diff))
    mse = np.mean(diff ** 2)
    nrmse = np.sqrt(mse) / data_range
    psnr = 20 * np.log10(data_range) - 10 * np.log10(mse)
    return psnr 


def extract_info(lines):
    psnr = [-1,-1]
    ssim = [-1,-1]
    i = 0 
    j = 0 
    for line in lines:
        if "PSNR" in line:
            psnr[i]=float(re.split('=|,', line)[1])
            i += 1
        if "SSIM" in line:
            ssim[j]=float(line.split()[-1])
            j += 1 
            
    return psnr, ssim 

def extract_zero_ratio(lines):
    # "zero ratio = x "
    for line in lines:
        if "zero ratio =" in line:
            return float(line.split()[-1])
    return -1
    

def gaussian_filtered(data, sigma):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(data, sigma=sigma)
  
# dims have the fastest dimension at the last
def run_instance(orig_file, dims, eb):
    thread_num = 8
    # exe_path = "/scratch/pji228/gittmp/posterization_mitigation/build/test/test_quantize_and_edt  "
    exe_path = exe_file_path
    # dims_str = str(len(dims)) + " " + " ".join([str(x) for x in dims])
    dims_str =  " ".join([str(x) for x in dims])
    out_dir = "./results/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    filename = os.path.basename(orig_file) 
    dec_file = out_dir + filename + ".dec" + str(eb)
    print(eb)
    compensated_file = out_dir + filename + ".comp"  + str(eb)
    
    command = f"{exe_path} -N {len(dims)} -d {dims_str} -i {orig_file} -m rel -e {eb} -q {dec_file} -c  {compensated_file} -t {thread_num} --use_rbf 0"
    print(command)
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print("results", result.stdout.split("\n"))
    try:
        os.remove(dec_file,)
        os.remove(compensated_file)
    except OSError:
        pass
    return  result.stdout
    

def run_process(i, j,orig_file, dims, eb ):
    stdout= run_instance(orig_file, dims, eb)
    psnr, ssim = extract_info(stdout.split("\n")) 
    return i, j,  psnr, ssim,  stdout


eb_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
sys.path.append("../../data/data_info/")
import datainfo
dataset_name = sys.argv[1]
cur_dataset = datainfo.DataSets[dataset_name]
file_num = cur_dataset.num_files
filenames = cur_dataset.file_names
print(filenames)


orig_ssim = np.zeros((file_num, len(eb_list)))
orig_psnr = np.zeros((file_num, len(eb_list)))

output = [[' ' for _ in range(len(eb_list))] for _ in range(file_num)]

post_dir1_ssim = np.zeros((file_num, len(eb_list)))
post_dir1_psnr = np.zeros((file_num, len(eb_list)))

post_dir2_ssim = np.zeros((file_num, len(eb_list)))
post_dir2_psnr = np.zeros((file_num, len(eb_list)))

post_dir3_ssim = np.zeros((file_num, len(eb_list)))
post_dir3_psnr = np.zeros((file_num, len(eb_list)))

post_3d_ssim = np.zeros((file_num, len(eb_list)))
post_3d_psnr = np.zeros((file_num, len(eb_list)))

post_6d_ssim = np.zeros((file_num, len(eb_list)))
post_6d_psnr = np.zeros((file_num, len(eb_list)))

post_avg_ssim = np.zeros((file_num, len(eb_list)))
post_avg_psnr = np.zeros((file_num, len(eb_list)))

zero_quant_ratio = np.zeros((file_num, len(eb_list))) 



from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=4) as executor:
    futures = []
    
    for i in range(file_num):
        for j in range(len(eb_list)):
            futures.append(executor.submit(run_process, i, j, cur_dataset.file_paths[i], cur_dataset.dims[::-1], eb_list[j]))
            

    # Collect results as they complete
    for future in futures:
        i, j, psnr, ssim, stdout = future.result()
        orig_psnr[i, j] = psnr[0]
        orig_ssim[i, j] = ssim[0]
        post_3d_ssim[i, j] = ssim[1]
        post_3d_psnr[i, j] = psnr[1]
        output[i][j] = stdout
        zero_quant_ratio[i, j] = extract_zero_ratio(stdout.split("\n"))
        print(i, j, psnr, ssim, stdout)


# write the results to a csv file

out_dir = "./"

out_dir="./"
out_log_name = out_dir + '/output_'+ dataset_name+'.log'
with open(out_log_name, 'w') as f:
    for i in range(file_num):
        for j in range(len(eb_list)):
            f.write(output[i][j])
            f.write('\n')
        
post_3d_psnr[post_3d_psnr==-1] = orig_psnr[post_3d_psnr==-1] 
post_3d_ssim[post_3d_ssim==-1] = orig_ssim[post_3d_ssim==-1]

psnr_tables = [orig_psnr,  post_3d_psnr]
tags = ['orig',  'post_3d']
psnr_pd_list = [] 
for i in range(len(psnr_tables)):
    table = pd.DataFrame(psnr_tables[i], columns=eb_list, index=filenames)
    agg_psnr = 10*np.log10(file_num/np.sum(10**(-0.1*psnr_tables[i]), axis=0))
    table.loc['agg_psnr'] = agg_psnr
    table.to_csv(out_dir + f'/psnr_{dataset_name}_{tags[i]}.csv', index=True, header=True, sep=',',float_format='%.10f')
    psnr_pd_list.append(table)

ssim_tables = [orig_ssim, post_3d_ssim]
tags = ['orig',  'post_3d']
ssim_pd_list = []
for i in range(len(psnr_tables)):
    table = pd.DataFrame(ssim_tables[i], columns=eb_list, index=filenames)
    agg_ssim = np.mean(ssim_tables[i], axis=0)
    table.loc['agg_ssim'] = agg_ssim
    table.to_csv( out_dir + f'/ssim_{dataset_name}_{tags[i]}.csv', index=True, header=True, sep=',',float_format='%.10f')
    ssim_pd_list.append(table)
    

zero_quant_ratio_table = pd.DataFrame(zero_quant_ratio, columns=eb_list, index=filenames)
zero_quant_ratio_table.to_csv(out_dir + f'/zero_quant_ratio_{dataset_name}.csv', index=True, header=True, sep=',',float_format='%.10f')

    

table_orig_psnr = pd.DataFrame(orig_psnr, columns=eb_list, index=filenames)
agg_psnr = 10*np.log10(file_num/np.sum(10**(-0.1*orig_psnr), axis=0))
table_orig_psnr.loc['agg_psnr'] = agg_psnr


table_orig_ssim = pd.DataFrame(orig_ssim, columns=eb_list, index=filenames)
agg_ssim = np.mean(orig_ssim, axis=0)
table_orig_ssim.loc['agg_ssim'] = agg_ssim



# # plot figures 

# fig, ax = plt.subplots(1, 2, figsize=(8, 4))
# colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k'] 
# markers = ['o', 's', 'v', '^', 'x', '+', 'D']
# lines = ['-', '--', '-.', ':']

# for i in range(len(psnr_tables)): 
#     ax[0].plot(eb_list, psnr_pd_list[i].loc['agg_psnr'], label=tags[i], color=colors[i], marker=markers[i], linestyle=lines[i])
#     ax[1].plot(eb_list, ssim_pd_list[i].loc['agg_ssim'], label=tags[i], color=colors[i], marker=markers[i], linestyle=lines[i])

# ax[0].set_xlabel('eb')
# ax[0].set_xscale('log')  # Set the x-axis to log scale
# ax[0].invert_xaxis()   
# ax[0].set_ylabel('agg_psnr')
# ax[0].legend()
# ax[0].grid()
# ax[1].set_xlabel('EB')
# ax[1].set_xscale('log')  # Set the x-axis to log scale
# ax[1].invert_xaxis()  # Invert the x-axis  
# ax[1].set_ylabel('SSIM ')
# ax[1].legend()
# ax[1].grid()
# fig.suptitle(f'{dataset_name} aggregation results')
# fig.savefig(out_dir + '/agg_'+dataset_name+'.pdf') 



