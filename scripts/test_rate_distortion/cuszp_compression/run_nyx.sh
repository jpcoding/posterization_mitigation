datastes="../../data/data/nyx_512x512x512/"
tag="nyx"
dims="512 512 512"
rm -rf ${tag}.log
source ../config.sh

for file in $datasets/*.f32; do
    filename=$(basename $file)
    echo "filename = ${filename}" >> ${tag}.log
    python run.py $file ${cuszp_path} ${dims} >> ${tag}.log
done