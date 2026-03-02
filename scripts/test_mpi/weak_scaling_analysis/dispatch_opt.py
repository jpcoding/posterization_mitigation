import os 
import sys 

def generate_sbatch(
    job_name,
    ntasks,
    nodes,
    ntasks_per_node,
    time_limit,
    partition,
    account,
    output_name,
    eb_list,
    data_dir,
    out_dir,
    mpi_dims,
    mpi_nps,
    origdims):
    content = f"""#!/bin/sh
#SBATCH --job-name=p_{job_name}        # Job name
#SBATCH --output={output_name}%j.out    # Standard output and error log
#SBATCH --error={output_name}%j.err       # Separate error log
#SBATCH --ntasks={ntasks}               # Number of MPI tasks
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --time={time_limit}              # Time limit (hh:mm:ss)
#SBATCH --partition={partition}          # Change this if needed
#SBATCH --account={account}          # Account name 
#SBATCH --exclusive                 # Request nodes exclusively
#SBATCH --mail-type ALL         # Send email when job starts/ends


test="../../../install/bin/test_mpi/test_approximate_parallel"

data_dir={data_dir}
out_dir={out_dir}
mkdir -p $out_dir 
iters=5
eb_list=({" ".join(str(x) for x in eb_list)})

for eb in ${{eb_list[@]}}
do
for i in $(seq 1 $iters)
do
echo "Running with rel_eb=$eb, iteration $i" 
mpirun -n {mpi_nps} $test --mpidims {" ".join(str(x) for x in mpi_dims )}  -m abs -e $eb --dir $data_dir  --outdir $out_dir --prefix vx --origdims {" ".join(str(x) for x in origdims )} --use_rbf 0 --local_edt 1 --local_quant 0
done
done 


"""
    return content


mpi_dims = [[8,8,8],[8,8,4],[8,4,4],[4,4,4]]
block_size = 512 
orig_dims = [] 
for i in range(len(mpi_dims)):
    orig_dims.append([block_size * x for x in mpi_dims[i]])
mpi_nps = [512, 256, 128, 64] 
# eb_list = [0.0001  , 0.001 ,0.01]
eb_list = [0.001]
value_range = 23.101325
abs_eb = [value_range * x for x in eb_list]
job_names = ["opt_8x8x8", "opt_8x8x4", "opt_8x4x4", "opt_4x4x4"] 
input_dir_names = ["vx_8x8x8", "vx_8x8x8", "vx_8x8x8", "vx_8x8x8"]
data_dir_root = "????/data/isotropic/" 
out_dir_root = "???/data/isotropic/output/" 
time_limit = "6:30:00"  
partition = "???" # change this to your partition
account = "???" # c
nodes = [16,8,4,2]
ntasks_per_node = [0,0,0,0]
for i in range(len(nodes)): 
    ntasks_per_node[i] = mpi_nps[i] // nodes[i]


# nodes = [2,2,2,2]
outputdir_list  = []
inputdir_list= [] 
for i in range(len(job_names)):
    outputdir_list.append(out_dir_root + job_names[i] + "/")
    inputdir_list.append(data_dir_root + job_names[i] + "/")

for i in range(len(job_names)):
    job_name = job_names[i]
    data_dir = data_dir_root + input_dir_names[i] + "/"
    out_dir = out_dir_root + job_name + "/"
    output_name = out_dir + job_name
    ntasks = nodes[i] * ntasks_per_node[i]
    sbatch_content = generate_sbatch(
        job_name,
        ntasks,
        nodes[i],
        ntasks_per_node[i],
        time_limit,
        partition,
        account,
        job_names[i],
        abs_eb,
        data_dir,
        out_dir,
        mpi_dims[i],
        mpi_nps[i],
        orig_dims[i])
    
    with open(f"{job_name}.sh", "w") as f:
        f.write(sbatch_content)

    



