datasets="../../data/data/miranda_256x384x384"
tag="miranda"
dims="384 384 256"
rm -rf ${tag}.log

for file in $datasets/*.f32; do
    filename=$(basename $file)
    echo "filename = ${filename}" >> ${tag}.log
    python run.py $file ${dims} >> ${tag}.log
done