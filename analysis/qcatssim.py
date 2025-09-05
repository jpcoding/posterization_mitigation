import numpy as np
from skimage.util import view_as_windows
from numba import njit, prange

import sys 

K1 = 0.01
K2 = 0.03

@njit
def SSIM_3d_window(orig, other):
    xMin = orig.min()
    xMax = orig.max()
    yMin = other.min()
    yMax = other.max()
    xMean = orig.mean()
    yMean = other.mean()
    var_x = orig.var()
    var_y = other.var()
    var_xy = ((orig - xMean) * (other - yMean)).mean()
    xSigma = np.sqrt(var_x)
    ySigma = np.sqrt(var_y)
    if (xMax - xMin) == 0:
        c1 = K1**2
        c2 = K2**2
    else:
        L = xMax - xMin
        c1 = (K1 * L) ** 2
        c2 = (K2 * L) ** 2

    c3 = c2 / 2
    luminance = (2 * xMean * yMean + c1) / (xMean**2 + yMean**2 + c1)
    contrast  = (2 * xSigma * ySigma + c2) / (xSigma**2 + ySigma**2 + c2)
    structure = (var_xy + c3) / (xSigma * ySigma + c3)
    ssim = luminance * contrast * structure
    return ssim

@njit(parallel=True)
def SSIM_3d_parallel(orig_windows, other_windows):
    z, y, x = orig_windows.shape[:3]
    total = 0.0
    print("orig_windows.shape", (z * y * x))
    for i in prange(z):
        for j in range(y):
            for k in range(x):
                total += SSIM_3d_window(orig_windows[i, j, k], other_windows[i, j, k])

    return total / (z * y * x)


def SSIM_3d(orig, other, window_shape=(7, 7, 7), stride=2):
    orig_windows = view_as_windows(orig, window_shape, step=stride)
    other_windows = view_as_windows(other, window_shape, step=stride)
    return SSIM_3d_parallel(orig_windows, other_windows)

file1 = sys.argv[1]
file2 = sys.argv[2]

orig = np.fromfile( file1, dtype=np.float32).reshape(256,384,384)
other = np.fromfile( file2, dtype=np.float32).reshape(256,384,384)
print(SSIM_3d(orig, other))

            
