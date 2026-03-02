import os
import sys
import time
import numpy as np
import ctypes

small_compressor = '/lcrc/project/ECP-EZ/jp/cuSZp/build/examples/bin/cuSZp'

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


def run_cusz(file, dims, eb, compressor ):
    compressor_path=compressor
    filename=os.path.basename(file)
    zfile=file+".cuszp"
    ofile=file+".cuszp.out"
    dim_string = " ".join([str(x) for x in dims])
    compress_command = f"{compressor_path}  -i {file} -t f32 -m plain  -eb rel  {eb} -x {zfile} -o {ofile} "
    print(compress_command, file=sys.stderr)
    os.system(compress_command)
    cr = os.path.getsize(file)*1.0/os.path.getsize(zfile)
    odata = np.fromfile(file,dtype=np.float32)
    ddata = np.fromfile(ofile, dtype=np.float32)
    psnr = verify(odata, ddata)
    os.remove(zfile)
    os.remove(ofile)
    return cr, psnr



eb_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

file = sys.argv[1]
compressor = sys.argv[2]
dims = [int(x) for x in sys.argv[3:]]

crs = []
psnrs = []

for eb in eb_list:
    cr,psnr = run_cusz(file, dims, eb, compressor)
    crs.append(cr)
    psnrs.append(psnr)