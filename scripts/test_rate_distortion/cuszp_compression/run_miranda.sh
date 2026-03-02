datasets="../../data/data/miranda_256x384x384/"
tag="miranda"
dims="384 384 256"
rm -rf ${tag}.log
source ../config.sh

for file in $datasets/*.f32; do
    filename=$(basename $file)
    echo "filename = ${filename}" >> ${tag}.log
    python run.py $file ${cuszp_path} ${dims} >> ${tag}.log
done