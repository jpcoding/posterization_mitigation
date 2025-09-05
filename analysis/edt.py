import numpy as np 
import os
import subprocess
import uuid 




def edt_model(input_file, eb, shape, user_rbf=0):
    
    print("shape: ", shape) 
    shape_str = ' '.join(map(str, shape)) 
    random_num = np.random.randint(0, 1000000) 
    shape_len = len(shape) 
    
    command = f'/scratch/pji228/gittmp/posterization_mitigation/build/test/test_quantize_and_edt -N {shape_len} -d {shape_str} \
                -m rel -e {eb} -i {input_file} \
                -q {random_num}.q -c {random_num}.c -t 8 --use_rbf {user_rbf}'  
    # os.system(command)
    print( command)
    result = subprocess.run(command,     stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    shell=True)
    # for line in result.stdout.split('\n'):
    #     if('PSNR' in line):
    #         print(line)
    #     if ('SSIM' in line):
    #         print(line)
    # result = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) 
    c_data = np.fromfile(f'{random_num}.c', dtype=np.float32).reshape(shape) 
    q_data = np.fromfile(f'{random_num}.q', dtype=np.float32).reshape(shape) 
    os.remove(f'{random_num}.c')
    os.remove(f'{random_num}.q') 
    return q_data, c_data, result.stdout


def edt(x, rel_eb, user_rbf=0 ):
    outfile = f'/tmp/{uuid.uuid4()}.f32' 
    x.astype(np.float32).tofile(outfile) 
    edt_result = edt_model(outfile, rel_eb, x.shape, user_rbf=user_rbf) 
    os.remove(outfile)
    return edt_result
