import os 
import sys
import time
import numpy as np
import matplotlib.pyplot as plt 
import ctypes
import subprocess
import pandas as pd 
import toml 


with open('config.toml', 'r') as f:
    config = toml.load(f.read())

exe_file_path = "../../../install/bin/test/direct_quantize"
# print(config)


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
    print("abs err={:.8G}".format(max_diff))
    mse = np.mean(diff ** 2)
    nrmse = np.sqrt(mse) / data_range
    psnr = 20 * np.log10(data_range) - 10 * np.log10(mse)
    return psnr 


def gaussian_filtered(data, sigma):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(data, sigma=sigma)


def direct_quantiz(file, rel_eb, dims, sigma = 1.0):
    base_name = os.path.basename(file) 
    result_dir = "./results/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    output = os.path.join(result_dir, base_name) 
    out_name = os.path.join(result_dir, base_name.split(".")[0] + f"{str(rel_eb)}")
    dfile_name = os.path.join(result_dir, base_name.split(".")[0] + f"{str(rel_eb)}" + ".out")
    command = f"{exe_file_path}  {file} {rel_eb} {out_name}" 
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    odata = np.fromfile(file, dtype=np.float32).reshape(dims)
    qdata = np.fromfile(dfile_name, dtype=np.float32).reshape(dims)
    orig_psnr = verify(odata, qdata) 
    orig_ssim = qcatssim(odata, qdata) 
    print("orig psnr={:.8G}".format(orig_psnr))
    print("orig ssim={:.8G}".format(orig_ssim))
    qdata = gaussian_filtered(qdata, sigma)
    post_psnr = verify(odata, qdata) 
    post_ssim = qcatssim(odata, qdata)
    max_error = np.max(abs(odata - qdata)) 
    os.remove(dfile_name)
    quant_name = os.path.join(result_dir, base_name.split(".")[0] + f"{str(rel_eb)}" + ".quant.i32")
    os.remove(quant_name)
    return  orig_psnr, orig_ssim, post_psnr, post_ssim, max_error, result.stdout
    
    
def run_process(i, j,orig_file, dims, eb, sigma):
    orig_psnr, orig_ssim, post_psnr, post_ssim, max_error, stdout = direct_quantiz(orig_file, eb, dims, sigma)
    return i, j, orig_psnr, orig_ssim, post_psnr, post_ssim, max_error, stdout 


eb_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
# eb_list = [1e-4,] 
sys.path.append("../../data/data_info/")
import datainfo
dataset_name = sys.argv[1]
cur_dataset = datainfo.DataSets[dataset_name]
file_num = cur_dataset.num_files
filenames = cur_dataset.file_names
print(filenames)

orig_ssim = np.zeros((file_num, len(eb_list)))
orig_psnr = np.zeros((file_num, len(eb_list)))

post_ssim = np.zeros((file_num, len(eb_list))) 
post_psnr = np.zeros((file_num, len(eb_list))) 
max_error = np.zeros((file_num, len(eb_list))) 
max_rel_error = np.zeros((file_num, len(eb_list))) 

output = [[' ' for _ in range(len(eb_list))] for _ in range(file_num)]


from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=16) as executor:
    futures = []
    
    for i in range(file_num):
        orig_file = os.path.join(cur_dataset.data_dir, filenames[i])
        temdata = np.fromfile(orig_file, dtype=np.float32)
        value_range = np.max(temdata) - np.min(temdata) 
        dims = cur_dataset.dims
        for j in range(len(eb_list)):
            eb = eb_list[j]
            futures.append(executor.submit(run_process, i, j, orig_file, dims[::-1], eb, 1.0))
    for future in futures:
        i, j, orig_psnr[i][j], orig_ssim[i][j], post_psnr[i][j], post_ssim[i][j], max_error[i][j], stdout = future.result()
        max_rel_error[i][j] = max_error[i][j] / value_range
        print(stdout)
        print("orig psnr={:.8G}".format(orig_psnr[i][j]))
        print("orig ssim={:.8G}".format(orig_ssim[i][j]))
        print("post psnr={:.8G}".format(post_psnr[i][j]))
        print("post ssim={:.8G}".format(post_ssim[i][j]))
        print("max error={:.8G}".format(max_error[i][j]))
        output[i][j] = stdout 
        



orig_psnr_table = pd.DataFrame(orig_psnr, columns=eb_list, index=filenames) 
agg_psnr = 10*np.log10(file_num/np.sum(10**(-0.1*orig_psnr), axis=0))
orig_psnr_table.loc['agg_psnr'] = agg_psnr
orig_psnr_table.to_csv(f"gaussian_{dataset_name}orig_psnr.csv")

post_psnr_table = pd.DataFrame(post_psnr, columns=eb_list, index=filenames)
agg_psnr = 10*np.log10(file_num/np.sum(10**(-0.1*post_psnr), axis=0))
post_psnr_table.loc['agg_psnr'] = agg_psnr
post_psnr_table.to_csv(f"gaussian_{dataset_name}post_psnr.csv")


orig_ssim_table = pd.DataFrame(orig_ssim, columns=eb_list, index=filenames)
agg_ssim = np.mean(orig_ssim, axis=0)
orig_ssim_table.loc['agg_ssim'] = agg_ssim
orig_ssim_table.to_csv(f"gaussian_{dataset_name}orig_ssim.csv")

post_ssim_table = pd.DataFrame(post_ssim, columns=eb_list, index=filenames)
agg_ssim = np.mean(post_ssim, axis=0)
post_ssim_table.loc['agg_ssim'] = agg_ssim
post_ssim_table.to_csv(f"gaussian_{dataset_name}post_ssim.csv")

max_error_table = pd.DataFrame(max_error, columns=eb_list, index=filenames)
max_error_table.to_csv(f"gaussian_{dataset_name}max_error.csv")

max_rel_error_table = pd.DataFrame(max_rel_error, columns=eb_list, index=filenames)
max_rel_error_table.to_csv(f"gaussian_{dataset_name}max_rel_error.csv")


    
    
    
    
    
    
    
    
    