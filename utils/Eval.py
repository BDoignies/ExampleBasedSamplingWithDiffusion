import os
import datetime
from collections import deque

from tensorboardX import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage
from scipy.spatial import distance_matrix


import torch

from data.Transforms import to_pointset_optimal_transport

class WriteSamples:
    def __init__(self, dir: str, format = "{it}.txt") -> None:
        self.dir = dir
        self.format = format

        os.makedirs(self.dir, exist_ok = True)
        os.makedirs(self.dir, exist_ok = True)

    def compute_and_write(self, writer, it, data, samples):
        samples = samples.reshape(samples.shape[0], np.prod(samples.shape[1:]))

        path = os.path.join(self.dir, self.format.format(it=it))
        np.savetxt(path, samples)

class PlotPointset:
    def __init__(self, save=False, dir="", format="points_{it}.png"):
        self.save = save
        self.dir = dir
        self.format = format

        # if self.save:
        #     os.makedirs(self.dir, exist_ok=True)
        #     os.makedirs(self.dir, exist_ok=True)

    def compute_and_write(self, name, it, writer: SummaryWriter, samples, names, props):    
        figcount = int(np.sqrt(samples.shape[0]))
        fig, axs = plt.subplots(figcount, figcount, subplot_kw={'aspect': 1})
        fig.set_size_inches((20, 20))

        for i, axsi in enumerate(axs):
            for j, ax in enumerate(axsi):
                idx = j + i * axs.shape[0]

                ax.set_aspect(1)
                ax.set_xticks(np.linspace(0, 1, samples.shape[2], endpoint=False))
                ax.set_yticks(np.linspace(0, 1, samples.shape[2], endpoint=False))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                # ax.grid(True)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

                ax.set_title(names[idx])
                ax.scatter(samples[idx, 0], samples[idx, 1])

        writer.add_figure(name, fig, it)
        # if self.save:
        #     fig.savefig(self.dir.format(it))

        plt.close(fig)

class EvalPointset:
    def __init__(
        self, 
        dir_   : str, 
        freq   : int, 
        ts     : list,
    ):
        self.dir = dir_
        self.freq = freq
        self.ts = ts
        self.create_metrics()
        
    def create_metrics(self):
        self.metrics = {}
        self.plot = PlotPointset()
        self.write = WriteSamples(os.path.join(self.dir, "eval", "samples"), format="{it}.txt")

    def compute(self, it: int, writer: SummaryWriter, model: torch.nn.Module, data: torch.tensor, force: bool =False):
        if it == 0:
            return 
        
        if it % self.freq == 0 or force:
            # Save only parameters 
            torch.save({
                "diffu": model.state_dict()
            }, os.path.join(writer.logdir, "model.ckpt"))
            
            for t in self.ts:
                model.set_num_timesteps(t)        
                for key in data:
                    cond  = data[key]["prop"].to(model.device)
                    shape = data[key]["data"].shape

                    pts = model.p_sample_loop(shape, img=None, cond=cond, with_tqdm=False, with_sampling=True)
                    pts = to_pointset_optimal_transport(pts.cpu().numpy())
                    # print(shape)

                    self.plot.compute_and_write(f"Plot/gen/{key}", it, writer, pts, data[key]['name'], data[key]['prop'])

                    pts = to_pointset_optimal_transport(data[key]["data"].cpu().numpy())
                    self.plot.compute_and_write(f"Plot/real/{key}", it, writer, pts, data[key]['name'], data[key]['prop'])
                    self.write.compute_and_write(writer, it, data, pts)


            model.reset_timesteps()
        