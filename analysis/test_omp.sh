thread_nums=(1 2 4 8 16 32 64)
iters=5
# test_log=test_omp.log
# rm -f $test_log
for tn in ${thread_nums[@]}; do
    echo "Testing with OMP_NUM_THREADS=$tn" 
for i in $(seq 1 $iters); do
echo "  Iteration $i" 
  OMP_NUM_THREADS=$tn ../build/test/test_quantize_and_edt -N 3 -d 100 500 500 -m rel -e 0.005 \
    -q /tmp/test.q -c /tmp/test.c  \
     -i $DATA/hurricane_100x500x500/Uf48.bin.f32 -t $tn >> breakdown.log

# OMP_NUM_THREADS=$tn /scratch/pji228/gittmp/SZp/install/bin/testfloat_compress_fastmode1 \
#     $DATA/hurricane_100x500x500/Uf48.bin.f32 \
#     64 0.005 float random 5 5 results.csv >> szp.log 
done 
done
