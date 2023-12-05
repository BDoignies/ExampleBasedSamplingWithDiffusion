import numpy as np
import h5py
import os

from tqdm import tqdm

def group_by_name(path, lvl=2):
    for _ in range(lvl):
        path = os.path.dirname(path)
    return os.path.basename(path)

class QMCDatabaseBuilder:
    def __init__(self, srcs, dest, properties, transform, group=group_by_name, loader=np.loadtxt):
        self.dest = dest
        self.loader = loader
        self.properties = properties
        self.transform = transform

        self.files = {}
        if isinstance(srcs, str):
            srcs = [srcs]
        
        for src in srcs:
            for root, dirs, files in os.walk(src, followlinks=True):
                for file in files:
                    fpath = os.path.join(root, file)

                    key = group(fpath)
                    if key not in self.files:
                        self.files[key] = []
                    
                    self.files[key].append(os.path.join(root, file))

    def build(self):
        with h5py.File(self.dest, "w") as output:
            for key in tqdm(self.files, total=len(self.files.keys())):
                shape_data_t = {}
                shape_data = {}
                shape_prop = {}

                for file in tqdm(self.files[key], leave=False):
                    data = self.loader(file)
                    N, D = data.shape
                    
                    prop = self.properties(key, data)
                    data_t = self.transform(data)

                    if N not in shape_data:
                        shape_data[N]   = []
                        shape_prop[N]   = []
                        shape_data_t[N] = []
                    
                    shape_data_t[N].append(data_t)
                    shape_data[N].append(data)
                    shape_prop[N].append(prop) 
                
                group = output.create_group(key)
                for s in shape_data:
                    shape_group = group.create_group(str(s))

                    shape_group.create_dataset("data"  , data=np.stack(shape_data[s]))
                    shape_group.create_dataset("data_t", data=np.stack(shape_data_t[s]))
                    shape_group.create_dataset("prop"  , data=np.stack(shape_prop[s]))
    