import numpy
import os

class DataInfo:
    name = None 
    data_dir = None
    ext = None
    file_paths = None
    file_names = None
    dims = None
    num_files = None
    dims_string = None
    dims = None
    N = None
    
    def __init__(self, data_dir, ext):
        self.data_dir = data_dir
        self.ext = ext
        self.file_paths, self.file_names = self.get_file_list()  
        self.num_files = len(self.file_paths) 
        self.file_num = self.num_files
        
    def set_dims (self, dims):
        self.dims = dims
        self.N = len(dims)
        if(self.N ==3): self.dims_string = f"{self.N} {dims[0]} {dims[1]} {dims[2]}"
        elif(self.N == 2): self.dims_string = f"{self.N} {dims[0]} {dims[1]}"
    
    def get_max_qoi(self, file):
        return numpy.max(numpy.fromfile(file, dtype=numpy.float32))
    
    def get_file_list(self):
        file_paths = []
        file_names = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(self.ext):
                    file_paths.append(os.path.join(root, file))
                    file_names.append(file)
        return file_paths, file_names
        
        

base_dir = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.abspath(os.path.join(base_dir, '..', 'data'))

miranda = DataInfo(f'{data_root}/miranda_256x384x384/', '.f32') 
miranda.set_dims([384,384,256])
miranda.name = 'miranda'

hurricane = DataInfo(f'{data_root}/huricane_100x500x500/', '.f32') 
hurricane.set_dims([500,500,100])
hurricane.name = 'hurricane'

nyx = DataInfo(f'{data_root}/nyx_512x512x512/', '.f32') 
nyx.set_dims([512,512,512])
nyx.name = 'nyx'

jhtdb_pressure = DataInfo(f'{data_root}/jhtdb_pressure_512x512x512/', '.f32')
jhtdb_pressure.set_dims([512,512,512])
jhtdb_pressure.name = 'jhtdb_pressure'



DataSets={}
DataSets['miranda'] = miranda
DataSets['hurricane'] = hurricane
DataSets['nyx'] = nyx
DataSets['jhtdb_pressure'] = jhtdb_pressure
    
        
    