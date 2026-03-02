import numpy as np
from matplotlib import pyplot as plt
from qcat import calculate_ssim
from edt import edt


def calculate_psnr(src_data, dec_data):
    data_range = np.max(src_data) - np.min(src_data)
    diff = src_data - dec_data
    mse = np.mean(diff ** 2)
    psnr = 20 * np.log10(data_range) - 10 * np.log10(mse)
    return psnr


def get_data(orig_path, shape, rel_eb, user_rbf=0):
    odata = np.fromfile(orig_path, dtype=np.float32).reshape(shape)
    ddata, cdata, output = edt(odata, rel_eb, user_rbf=user_rbf)
    orig_psnr = calculate_psnr(odata, ddata)
    post_psnr = calculate_psnr(odata, cdata)
    orig_ssim = calculate_ssim(odata, ddata)
    post_ssim = calculate_ssim(odata, cdata)
    return {
        'odata': odata,
        'ddata': ddata,
        'post': cdata,
        'orig_psnr': orig_psnr,
        'post_psnr': post_psnr,
        'orig_ssim': orig_ssim,
        'post_ssim': post_ssim,
        'output': output,
    }


data_path = '../data/hurricane_100x500x500/Wf48.bin.f32'
results_high = get_data(data_path, (100, 500, 500), 0.01, 0)
results_med  = get_data(data_path, (100, 500, 500), 0.003, 0)
results_low  = get_data(data_path, (100, 500, 500), 0.001, 0)

## compare orig with the decompressed data
locations = (50, slice(275, 325), slice(275, 325))
results = [results_high, results_med, results_low][::-1]

figs, axs = plt.subplots(3, 3, figsize=(9, 10))
d_max = np.max(results_high['odata'][locations])
d_min = np.min(results_high['odata'][locations])
tag = [' $A$', ' $B$', ' $C$']
for i in range(3):
    odata = results[i]['odata']
    ddata = results[i]['ddata']
    post  = results[i]['post']
    orig_error      = odata - ddata
    predicted_error = post  - ddata
    e_max = np.max(orig_error[locations])
    e_min = np.min(orig_error[locations])
    local_orig_psnr = calculate_psnr(odata[locations], ddata[locations])
    local_post_psnr = calculate_psnr(odata[locations], post[locations])
    local_orig_ssim = calculate_ssim(odata[locations], ddata[locations])
    local_post_ssim = calculate_ssim(odata[locations], post[locations])
    axs[i, 0].imshow(odata[locations], cmap='coolwarm', vmin=d_min, vmax=d_max)
    axs[i, 0].set_title('Original Data')
    axs[i, 1].imshow(ddata[locations], cmap='coolwarm', vmin=d_min, vmax=d_max)
    axs[i, 1].set_title(f'Quantized Data{tag[i]}\n PSNR={local_orig_psnr:.2f}, SSIM={local_orig_ssim:.2f}')
    axs[i, 2].imshow(post[locations], cmap='coolwarm', vmin=d_min, vmax=d_max)
    axs[i, 2].set_title(f'Compensated Data{tag[i]}\n PSNR={local_post_psnr:.2f}, SSIM={local_post_ssim:.2f}')

plt.tight_layout()
cbar = figs.colorbar(axs[0, 0].images[0], ax=axs, orientation='vertical', fraction=0.05, pad=0.005, aspect=80)
plt.savefig('case_study_hurricane.pdf', bbox_inches='tight')
