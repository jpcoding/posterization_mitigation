import os 
import numpy as np 
import sys 
import datainfo 
import gdown 



dataset_names = ['miranda', 'hurricane', 'nyx','s3d']
urls =[
    'https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/Miranda/SDRBENCH-Miranda-256x384x384.tar.gz',
    'https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/Hurricane-ISABEL/SDRBENCH-Hurricane-ISABEL-100x500x500.tar.gz',
    'https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/EXASKY/NYX/SDRBENCH-EXASKY-NYX-512x512x512.tar.gz',
    "https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/S3D/SDRBENCH-S3D.tar.gz"
]



# dataset_names = ['jhtdb_pressure']
# urls =[
#     "https://drive.google.com/uc?id=1bWmZx1tHVYCjg2ACQDbS6rbf0X4CPC53"
# ]

dataset_names = ['s3d']
urls =[
    "https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/S3D/SDRBENCH-S3D.tar.gz"
]


for  dataset in dataset_names:
    i = dataset_names.index(dataset)
    print(datainfo.DataSets[dataset].data_dir)
    cur_dir = os.path.join(datainfo.DataSets[dataset].data_dir)    
    if not os.path.exists(cur_dir):
        os.makedirs(cur_dir)
    # download the data using the url 
    # and save it to the current directory
    if(dataset != 'jhtdb_pressure'): 
        command = f"cd {cur_dir} && wget {urls[i]} && tar -xvf  *.tar.gz -C . --strip-components=1  && rm *.tar.gz && rm -rf *log10* "
        os.system(command)
        # if miranda, convert f64 to f32 
        if(dataset == 'miranda' ):
            files = os.listdir(cur_dir)
            files = [os.path.join(cur_dir, file) for file in files]
            for file in files:
                if file.endswith('.d64'):
                    data = np.fromfile(file, dtype=np.float64)
                    data = data.astype(np.float32)
                    data.tofile(file.replace('.d64', '.f32'))
                    os.remove(file)
        
        if (dataset == 's3d'):
            # only extract stat_planar.1.1000E-03.field.d64 and convert it to f32
            # then extract the eleven fields names them as firld_0.f32, field_1.f32, ..., field_10.f32
            files = os.listdir(cur_dir)
            files = [os.path.join(cur_dir, file) for file in files]
            for file in files:
                if file.endswith('.d64'):
                    if 'stat_planar.1.1000E-03.field.d64' in file:
                        data = np.fromfile(file, dtype=np.float64)
                        data = data.reshape((512, 512, 512, 11))
                        for i in range(11):
                            field_data = data[:, :, :, i]
                            field_data = field_data.astype(np.float32)
                            field_data.tofile(file.replace('.d64', f'_field_{i}.f32'))
                        os.remove(file)
            


    else:
        output = cur_dir + '/jhtdb_pressure_512x512x512.tar.gz' 
        print(output)
        gdown.download(urls[i], output, quiet=False)
        command = f"cd {cur_dir} && tar -xvf  jhtdb_pressure_512x512x512.tar.gz -C . --strip-components=1  && rm jhtdb_pressure_512x512x512.tar.gz  "
        os.system(command)


        



