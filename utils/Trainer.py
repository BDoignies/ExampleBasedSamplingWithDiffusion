import os
import time

import torch
import numpy as np

from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader
from ema_pytorch import EMA

def cycle(dataset):
    while True:
        for data in dataset:
            yield data

class Trainer:
    def __init__(self,
        path,
        model,
        dataset,
        diffusion,
        train_params,
        eval,
        device = torch.device('cuda')
    ):
        self.device = device
        self.train_params = train_params
        self.eval = eval

        self.dataset = cycle(DataLoader(
            dataset, 
            batch_size = train_params["batch_size"], 
            shuffle = True, 
            pin_memory = True,
            num_workers = 10,
            prefetch_factor = 1
        ))
        self.num_timesteps = diffusion.betas.shape[0]

        self.model = model.to(device)
        self.diffusion = diffusion.to(device)
        self.ema = EMA(
            self.diffusion, 
            beta        = train_params["ema_decay"], 
            update_every= train_params["ema_update_every"]
        ).to(device)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=train_params["lr"])
        self.writer = SummaryWriter(path)
        # torch.backends.cudnn.benchmark = True

    def save(self, path):
        data = {
            # 'model': self.model.state_dict(), 
            'diff': self.diffusion.state_dict(), # Includes model 
            'optim': self.optim.state_dict(), 
            'ema': self.ema.state_dict()
        }
        torch.save(data, path)

    @torch.no_grad()
    def sample(self, noise, cond=None):
        self.model.eval()
        self.diffusion.eval()
        self.ema.eval()

        data = self.diffusion.p_sample_loop(None, noise, cond).detach().cpu().numpy()

        self.ema.train()
        self.diffusion.train()
        self.model.train()

        return data

    def sample_with_grad(self, noise, cond=None):
        # No TQDM, No Sampling
        samples = self.diffusion.p_sample_loop(None, noise, cond, False, False)
        return samples
    
    def train(self, it, withtqdm=True, to=1e8):
        its = range(it)
        if withtqdm:
            from tqdm import tqdm
            its = tqdm(its)

        tinit = time.time()
        end = False

        for i in its:
            data = next(self.dataset)
            # Loss + optim step
            self.optim.zero_grad()

            # Data contains a mapping dimension to samples 
            loss = 0
            
            # Same timestep for evey scale
            bs = data[list(data.keys())[0]]["data"].shape[0]
            ts = torch.randint(0, self.num_timesteps, (bs, )).to(self.device)
            
            for scale, scale_data in data.items():
                d = scale_data["data"].to(self.device)
                c = scale_data["prop"].to(self.device)
                
                scale_loss = self.diffusion(d, ts, x_cond = c)
                self.writer.add_scalar(f"Loss/{scale}", scale_loss / int(scale), i)
                
                loss += scale_loss

            self.writer.add_scalar(f"Loss/all", loss, i)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            
            self.optim.step()

            # Exponential moving average pass
            self.ema.update()

            runtime = (time.time() - tinit) / 60
            end = runtime > to
            
            with torch.no_grad():
                self.diffusion.eval()
                self.eval.compute(it=i, writer=self.writer, model=self.diffusion, data=data, force=(end or (i == it - 1)))
                self.diffusion.train()

            if end:
                break