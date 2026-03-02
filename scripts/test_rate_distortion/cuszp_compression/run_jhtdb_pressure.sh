datasets="../../data/data/jhtdb_pressure_512x512x512/"
tag="jhtdb_pressure"
dims="512 512 512"
rm ${tag}.log
source ../config.sh

for file in $datasets/*.f32; do
    filename=$(basename $file)
    echo "filename = ${filename}" >> ${tag}.log
    python run.py $file ${cuszp_path}  ${dims} >> ${tag}.log
done