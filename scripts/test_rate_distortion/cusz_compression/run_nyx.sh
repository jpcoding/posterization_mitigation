datasets="../..//data/data/nyx_512x512x512/"
tag="nyx"
dims="512 512 512"
rm -rf ${tag}.log

for file in $datasets/*.f32; do
    filename=$(basename $file)
    echo "filename = ${filename}" >> ${tag}.log
    python run.py $file ${dims} >> ${tag}.log
done