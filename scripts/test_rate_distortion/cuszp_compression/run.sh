sh run_miranda.sh
sh run_hurricane.sh
sh run_nyx.sh 
sh run_jhtdb_pressure.sh

datasets=('hurricane' 'miranda' 'nyx' 'jhtdb_pressure')
for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"
    python get_cr_files.py ${dataset}.log >${dataset}_cuszp.log 
done