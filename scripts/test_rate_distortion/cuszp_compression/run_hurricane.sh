datasets="../../data/data/huricane_100x500x500/"
tag="hurricane"
dims="500 500 100"
rm -rf ${tag}.log
source ../config.sh
for file in $datasets/*.f32; do
    filename=$(basename $file)
    echo "filename = ${filename}" >> ${tag}.log
    python run.py $file ${cuszp_path} ${dims} >> ${tag}.log
done