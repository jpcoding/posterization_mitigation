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

with open('../test_rate_distortion/config.toml', 'r') as f:
    config = toml.load(f.read())

exe_file_path = "../../install/bin/test/test_quantize_and_edt" 
# print(config)


# def qcatssim(orig:np.ndarray[np.float32], decompressed:np.ndarray[np.float32]):
#     dims = np.array(orig.shape)[::-1]
#     # print(dims)
#     lib = ctypes.CDLL('/home/pji228/workspace/git/draft/artifact/compress_scripts/aramco_ssim/qcatssim_py.dylib')
#     lib.calculateSSIM.restype = ctypes.c_double
#     lib.calculateSSIM.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32),
#                               np.ctypeslib.ndpointer(dtype=np.float32),
#                               np.ctypeslib.ndpointer(dtype=np.int32),
#                               ctypes.c_int]
#     result =lib.calculateSSIM(
#         orig, 
#         decompressed,
#         dims.astype(np.int32), 
#         ctypes.c_int(dims.size))
#     return result 



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
    return 0,0 


def extract_zero_ratio(lines):
    # "zero ratio = x "
    for line in lines:
        if "zero ratio =" in line:
            return float(line.split()[-1])
    return -1

def extract_time (lines):
    # "Time = x "
    for line in lines:
        if "compensation time = " in line:
            return float(line.split()[-1])
    return -1
    

def gaussian_filtered(data, sigma):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(data, sigma=sigma)
  
# dims have the fastest dimension at the last
def run_instance(orig_file, dims, eb,thread_num):
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
    
    command = f"{exe_path} -N {len(dims)} -d {dims_str} -i {orig_file} -m rel -e {eb} -q {dec_file} -c  {compensated_file} -t {thread_num} --use_rbf 0 --no_ssim 1"
    print(command)
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print("results", result.stdout.split("\n"))
    try:
        os.remove(dec_file,)
        os.remove(compensated_file)
    except OSError:
        pass
    return  result.stdout
    

def run_process(i, j,orig_file, dims, eb, threads ):
    stdout= run_instance(orig_file, dims, eb,threads)
    psnr, ssim = extract_info(stdout.split("\n")) 
    return i, j,  psnr, ssim,  stdout


# eb_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
eb_list = [1e-4, 1e-3,  1e-2]
# eb_list = [1e-3]
sys.path.append('../data/data_info/')
import datainfo
dataset_name = sys.argv[1]
cur_dataset = datainfo.DataSets[dataset_name]
file_num = cur_dataset.num_files
filenames = cur_dataset.file_names
print(filenames)
time_iters = 5
thread_nums = [1, 2, 4, 8, 16, 32, 64]
# thread_nums = [16,32]
time_array = np.zeros((len(thread_nums), time_iters, file_num, len(eb_list))) 
# output = [[' ' for _ in range(len(eb_list))] for _ in range(file_num)]
output = [[[[' ' for _ in range(len(eb_list))] for _ in range(file_num)] for _ in range(time_iters)] for _ in range(len(thread_nums))] 

from concurrent.futures import ProcessPoolExecutor
with ProcessPoolExecutor(max_workers=1) as executor:
    for t  in range(len(thread_nums)): 
        for k in range(time_iters):
            futures = []
            for i in range(file_num):
                for j in range(len(eb_list)):
                    futures.append(executor.submit(run_process, i, j, cur_dataset.file_paths[i], cur_dataset.dims[::-1], eb_list[j], thread_nums[t]))

            for future in futures:
                i, j, psnr, ssim, stdout = future.result()
                output[t][k][i][j] = stdout
                time_array[t, k, i, j] = extract_time(stdout.split("\n")) 
                print(i, j, psnr, ssim, stdout)


np.save( './time_array_' + dataset_name + '.npy', time_array)

out_dir="./"
out_log_name = out_dir + '/output_'+ dataset_name+'.log'
with open(out_log_name, 'w') as f:
    for t in range(len(thread_nums)): 
        for k in range(time_iters): 
            for i in range(file_num):
                for j in range(len(eb_list)):
                    f.write(output[t][k][i][j])
                    f.write('\n')

#np.savetxt(out_dir + '/time_array_' + dataset_name + '.txt', time_array)

# omit max and min and get avg 



