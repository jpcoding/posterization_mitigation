grep SSIM slurm_opt.out | cut -d = -f 2 > ssim.txt 
grep PSNR slurm_opt.out | cut -d :  -f 2  > psnr.txt
python plot.py