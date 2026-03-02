import os
import sys
import numpy as np
import toml 

with open('../configs.toml', 'r') as f:
    configs=toml.load(f.read()) 

cusz_path = configs['cusz_path']

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


def run_cusz(file, dims, eb):
    compressor_path=cusz_path
    dim_string = "-".join([str(x) for x in dims])
    compress_command = f"{compressor_path} -z -i {file} -t f32 -m rel -e {eb} -l {dim_string} --report cr "
    print(compress_command,file = sys.stderr)
    os.system(compress_command)
    decompress_command = f"{compressor_path} -i {file}.cusza -x --report time --compare {file}"
    print(decompress_command,file = sys.stderr)
    os.system(decompress_command)
    cr = os.path.getsize(file)*1.0/os.path.getsize(file+".cusza")
    odata = np.fromfile(file,dtype=np.float32)
    ddata = np.fromfile(file+".cuszx", dtype=np.float32)
    psnr = verify(odata, ddata)
    os.remove(file+".cusza")
    os.remove(file+".cuszx")
    return cr, psnr



eb_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

file = sys.argv[1]
dims = [int(x) for x in sys.argv[2:]]
crs = []
psnrs = []

for eb in eb_list:
    cr,psnr = run_cusz(file, dims, eb)
    crs.append(cr)
    psnrs.append(psnr)

print("cusz_psnr = ", psnrs)
print("cusz_cr = ", crs)