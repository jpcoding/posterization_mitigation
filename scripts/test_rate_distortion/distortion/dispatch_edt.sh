dirs=(  'nyx'  'jhtdb_pressure'    'miranda'  'hurricane' )
for i in "${dirs[@]}"
do
    echo "Running $i"
    python run_edt_3d.py ${i} 
done

