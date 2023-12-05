import numpy as np
import torch
import h5py
import re

class QMCDataset(torch.utils.data.Dataset):
    __DATA_NAME__ = "data_t"
    __PROP_NAME__ = "prop"

    def __init__(self, path, samplers=['.*'], scales=['.*'], cond_index=Ellipsis, mask_index=[[]]):
        self.path = path
        self.dataset = None

        # print(mask_index)
        self.cond_index = cond_index
        self.mask_index = mask_index
        self.samplers = []
        self.samplers_scales = {}
        self.samplers_start_idx = [0]

        if not isinstance(scales, list):
            scales = [scales]

        if not isinstance(samplers, list):
            samplers = [samplers]

        scales = [str(s) for s in scales]
        

        # Compute indices and samplers
        with h5py.File(self.path, 'r') as file:        
            for key in file.keys():
                valid = False
                for sampler in samplers:
                    valid = valid or re.search(sampler, key)
                
                if valid:
                    self.samplers.append(key)

            for sampler in self.samplers:
                count = int(1e15)
                for key in file[sampler]:
                    valid = False
                    for scale in scales:
                        valid = valid or re.search(scale, key)
                    
                    if valid:
                        if sampler not in self.samplers_scales:
                            self.samplers_scales[sampler] = []

                        self.samplers_scales[sampler].append(f"{key}")
                        count = min(count, file[sampler][key][self.__DATA_NAME__].shape[0])

                self.samplers_start_idx.append(
                    self.samplers_start_idx[-1] + count
                )
        
    def __len__(self):
        return self.samplers_start_idx[-1]

    def find_sampler(self, index):
        i = 0
        while self.samplers_start_idx[i] <= index:
            i = i + 1
        return self.samplers[i - 1], index - self.samplers_start_idx[i - 1]
        
    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.path, 'r')
        
        sampler, index = self.find_sampler(index)
        if len(self.mask_index) > 0:
            mask = self.mask_index[np.random.randint(0, len(self.mask_index))]
        else:
            mask = Ellipsis

        scales = {}
        for scale in self.samplers_scales[sampler]:
            scales[scale] = {
                "name": sampler,
                "data": self.dataset[sampler][scale][self.__DATA_NAME__][index].astype(np.float32),
                "prop": self.dataset[sampler][scale][self.__PROP_NAME__][index][self.cond_index].astype(np.float32)
            }
            scales[scale]["prop"][mask] = 0
            # scales[scale]["prop"][mask] = -1.

        # if len(scales) == 1:
        #     return scales[next(scales.keys())]
        return scales
