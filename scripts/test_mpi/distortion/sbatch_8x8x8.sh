#!/bin/sh
#SBATCH --job-name=opt_job     # Job name
#SBATCH --output=slurm_opt.out    # Standard output and error log
#SBATCH --error=slurm_opt.err       # Separate error log
#SBATCH --ntasks=512           # Number of MPI tasks
#SBATCH --nodes=16              # Number of nodes
#SBATCH --ntasks-per-node=32   # Number of tasks per node
#SBATCH --time=5:00:00              # Time limit (hh:mm:ss)
#SBATCH --partition={???partition}          # Change this if needed
#SBATCH --account={???account}          # Change this if needed
#SBATCH --mail-type ALL         # Send email when job starts/ends
#SBATCH --exclusive


test="../../../install/bin/test_mpi/test_approximate_parallel"
merger_file="../../../install/bin/test_mpi/test_merge_file"
ssim_mpi="../../../install/bin/test_mpi/test_ssim_mpi_merged_file"

data_dir=???/isotropic_bin/vx_8x8x8/
out_dir=???/isotropic_bin/output_8x8x8/
mkdir -p $out_dir

mpirun -n 512 $merger_file --mpidims 8 8 8 --dir $data_dir --prefix vx --suffix .f32 --orig_dims 4096 4096 4096 --out merged.orig.f32

iters=1
eb_list=( 0.0001  0.0005  0.001 0.005 0.01 )
for eb in ${eb_list[@]}
do
    for i in $(seq 1 $iters)
    do
        mpirun -n 512 $test --mpidims 8 8 8 -m rel -e  $eb --dir $data_dir  --outdir $out_dir --prefix vx --origdims 4096 4096 4096 --use_rbf 0 --local_edt 1 --local_quant 0
    done
    mpirun -n 512 $merger_file --mpidims 8 8 8 --dir $out_dir --prefix vx --suffix .decomp.f32 --orig_dims 4096 4096 4096 --out merged.decompressed.f32
    mpirun -n 512 $merger_file --mpidims 8 8 8 --dir $out_dir --prefix vx --suffix .post3d.f32 --orig_dims 4096 4096 4096 --out merged.post.f32
    mpirun -n 512 $ssim_mpi 8 8 8 merged.orig.f32   merged.decompressed.f32  4096 4096 4096
    mpirun -n 512 $ssim_mpi 8 8 8 merged.orig.f32  merged.post.f32  4096 4096 4096
done
