from turtle import forward
import numpy as np

import torch
from torch import nn, randint
import torch.nn.functional as F
import torchvision.transforms.functional as T



# Some aliases to maintain same names
NIN   = nn.Linear   # nn.py lines 55-62
Dense  = nn.Linear  # nn.py lines 65-73
Conv2d = nn.Conv2d  # nn.py lines 76-87

nonlinearity = F.silu # unet.py lines 11-12

class IdentityWithEmbedding(nn.Identity):
    def forward(self, x, *args):
        return x

class SequentialWithEmbedding(nn.Sequential):
    def forward(self, x, *args):
        for module in self._modules.values():
            x = module(x, *args)
        return x

class SequentialWithCatAndEmbedding(nn.Sequential):
    def __init__(self, *modules, last_module=None):
        super().__init__(*modules)
        self.last = last_module

    def forward(self, x, x2, emb):
        for module in self._modules.values():
            x = module(torch.cat((x, x2), 1), emb)
        if self.last:
            x = self.last(x)
        return x


# Conv2d but with padding = 'same' as in tensorflow
# Formulas only work when stride=1 and dilatation=1
class Conv2dSame(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), bias=True, padding_layer=torch.nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size[0] // 2
        kb = ka - 1 if kernel_size[0] % 2 == 0 else ka
        kc = kernel_size[1] // 2
        kd = kc - 1 if kernel_size[1] % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka,kb,kc,kd)),
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        )
    def forward(self, x):
        return self.net(x)

# class NormalizeLayer(nn.GroupNorm):
#     def __init__(self, num_channels: int) -> None:
#         super().__init__(min(32, num_channels), num_channels, 0.000001)

class NormalizeLayer(nn.GroupNorm):
    def __init__(self, num_channels: int) -> None:
        super().__init__(1, num_channels, 0.000001)

# unet.py lines 15-23
class UpsampleLayer(nn.Module):
    def __init__(self, num_channels, use_conv=True) -> None:
        super().__init__()
        self.use_conv = use_conv

        if self.use_conv:
            self.conv = Conv2dSame(
                in_channels=num_channels,
                out_channels=num_channels,
            )

    def forward(self, x: torch.Tensor, emb: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = T.resize(x, [H * 2, W * 2], T.InterpolationMode.NEAREST)

        if self.use_conv:
            x = self.conv(x)

        return x

# unet.py lines 26-34
class DownsampleLayer(nn.Module):
    def __init__(self, num_channels, use_conv=True) -> None:
        super().__init__()
        self.use_conv = use_conv

        if self.use_conv:
            # Same as upsample, except for stride
            # TODO : Use a pad = same. However this
            #        should work work fine with powers of two
            #        shapes 
            self.layer = Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=(3, 3),
                stride = 2,
                padding = 1
            )
        else:
            # TODO : Compute padding for 
            # Avg pooling to match tf behaviour ...
            raise ValueError(
                    "Only use_conv=True is supported"
                    " in Downsampling layer"
                )
    
    def forward(self, x: torch.Tensor, emb: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if self.use_conv:
            return self.layer(x)
        return None

class ResnetBlock(nn.Module):
    def __init__(self, 
        num_channels: int, emb_dim: int, 
        out_channels: int = None, 
        dropout = 0.3, conv_shortcut = False) -> None:
        super().__init__()

        self.shortcut = conv_shortcut
        if out_channels is None:
            out_channels = num_channels

        self.norm1 = NormalizeLayer(num_channels)
        self.conv1 = Conv2dSame(num_channels, out_channels)
        self.norm2 = NormalizeLayer(out_channels)
        self.conv2 = Conv2dSame(out_channels, out_channels)
        self.dense = Dense(emb_dim, out_channels)
        self.dropout = nn.Dropout(dropout)

        if num_channels != out_channels:
            if self.shortcut:
                self.out_layer = Conv2dSame(num_channels, out_channels)
            else:
                self.out_layer = NIN(num_channels, out_channels)
        else:
            self.out_layer = nn.Identity()

    def forward(self, x, emb, cond: torch.Tensor):        
        h = x
        h = nonlinearity(self.norm1(h))
        h = self.conv1(h)

        h += self.dense(nonlinearity(emb))[..., None, None]

        h = nonlinearity(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        
        #  Applyed per channel 
        # reshape (B, C, H, W) => (B, H, W, C) 
        x = torch.transpose(torch.transpose(x, 1, 2), 2, 3)
        x = self.out_layer(x)
        x = torch.transpose(torch.transpose(x, 3, 2), 2, 1)
        
        # Re-reshape in the appropriate format
        return x + h
        
class AttentionBlock(nn.Module):
    def __init__(self, num_channels: int, cond_embedding: int = None) -> None:
        super().__init__()
        self.scale = int(num_channels) ** (-0.5)
        self.norm  = NormalizeLayer(num_channels)
        
        if cond_embedding is None or cond_embedding == 0:    
            self.has_cond = False
            cond_embedding = num_channels
        else:
            self.has_cond = True
        
        self.qnin  = NIN(cond_embedding, num_channels)
        self.knin  = NIN(num_channels, num_channels)
        self.vnin  = NIN(num_channels, num_channels)
        self.nin   = NIN(num_channels, num_channels)


    def forward(self, x: torch.Tensor, emb: torch.Tensor, cond: torch.Tensor):
        B, C, H, W = x.shape
        
        h  = self.norm(x)
        # reshape (B, C, H, W) => (B, H, W, C) 
        h = torch.transpose(torch.transpose(h, 1, 2), 2, 3)
        if self.has_cond:
            cond = torch.transpose(torch.transpose(cond, 1, 2), 2, 3)
            q = self.qnin(cond)    
        else:
            q = self.qnin(h)
        
        k = self.knin(h)
        v = self.vnin(h)

        w = torch.einsum("bhwc,bHWc->bhwHW", q, k) * self.scale
        w = w.view([B, H, W, H * W])
        w = F.softmax(w, -1)
        w = w.view([B, H, W, H, W])

        h = torch.einsum("bhwHW,bHWc->bhwc", w, v)
        h = self.nin(h)

        # reshape (B, H, W, C) => (B, C, H, W) 
        h = torch.transpose(torch.transpose(h, 3, 2), 2, 1)
        return x + h


# nn.py lines 90-109
def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    assert timesteps.ndim == 1

    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(0, half_dim) * -emb).to(timesteps.device)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.concat([torch.sin(emb), torch.cos(emb)], axis=1)
    
    # @WARNING: Check for paddig here
    # if embedding_dim % 2 == 1:
    #    emb = F.pad(emb, [[0, 0], [0, 1]])
    assert emb.shape == torch.Size([timesteps.shape[0], embedding_dim])
    return emb