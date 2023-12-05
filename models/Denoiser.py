import torch
import math

import torch.nn as nn
from models.Layers import (
    Dense, 
    AttentionBlock, 
    Conv2dSame, 
    ResnetBlock, 
    SequentialWithEmbedding, 
    IdentityWithEmbedding, 
    DownsampleLayer, 
    NIN,
    UpsampleLayer, 
    NormalizeLayer, 
    get_timestep_embedding, 
    nonlinearity
)

class DenoiserModel(nn.Module):
    def __init__(self, 
        num_channels: int, ch: int, out_ch: int, ch_mult: list, 
        num_res: int, attn_layers: list, attn_middle: bool,
        dropout=0.1, resamp_with_conv: bool=True,

        cond_model: nn.Module = nn.Identity, 
        cond_features: int = None, 
        cond_emb_dim: int = None
    ) -> None:
        super().__init__()
        self.ch = ch
        
        embdim = ch * 4
        if cond_emb_dim is None:
            cond_emb_dim = 0

        ch_mult = [ch * c for c in ch_mult]
        
        # Embedding layers
        self.dense1 = Dense(embdim, embdim)
        self.dense2 = Dense(embdim, embdim)

        # Encoder part of the unet
        self.encoder_layers = []
        self.downsamp_layers = []
        for i_level in range(len(ch_mult)):
            prev_ch = ch_mult[0] if i_level == 0 else ch_mult[i_level - 1]
            
            level_modules = []
            for _ in range(num_res):
                block_modules = [
                    ResnetBlock(prev_ch + cond_emb_dim, embdim, ch_mult[i_level], dropout)
                ]

                if i_level in attn_layers:
                    block_modules.append(
                        AttentionBlock(ch_mult[i_level], cond_emb_dim)
                    )
                level_modules.append(SequentialWithEmbedding(*block_modules))
                prev_ch = ch_mult[i_level]

            self.encoder_layers.append(nn.ModuleList(level_modules))
            if i_level != len(ch_mult) - 1:
                self.downsamp_layers.append(
                    DownsampleLayer(prev_ch, resamp_with_conv)
                )
            else:
                self.downsamp_layers.append(
                    IdentityWithEmbedding()
                )

        self.encoder_layers  = nn.ModuleList(self.encoder_layers)
        self.downsamp_layers = nn.ModuleList(self.downsamp_layers)

        if attn_middle:
            self.middle = SequentialWithEmbedding(
                ResnetBlock(ch_mult[-1], embdim, dropout=dropout),
                AttentionBlock(ch_mult[-1], cond_emb_dim),
                ResnetBlock(ch_mult[-1], embdim, dropout=dropout)
            )
        else:
            self.middle = SequentialWithEmbedding(
                ResnetBlock(ch_mult[-1], embdim, dropout=dropout),
                # AttentionBlock(ch_mult[-1], cond_emb_dim),
                ResnetBlock(ch_mult[-1], embdim, dropout=dropout)
            )

        # Decoder part of the Unet
        self.decoder_layers = []
        self.upsamp_layers = []

        for i_level in reversed(range(len(ch_mult))):
            enc_ch = ch_mult[i_level]

            level_modules = []
            prev_ch = ch_mult[i_level]
            for i_block in range(num_res):
                cat_channels = 2 * enc_ch + cond_emb_dim
                block_modules = [
                    ResnetBlock(cat_channels, embdim, prev_ch, dropout=dropout)
                ]

                if i_level in attn_layers:
                    block_modules.append(
                        AttentionBlock(prev_ch, cond_emb_dim)
                    )
                
                level_modules.append(SequentialWithEmbedding(*block_modules))
                prev_ch = ch_mult[i_level - 1] if i_level > 0 else ch_mult[0]
            
            self.decoder_layers.append(nn.ModuleList(level_modules))
            
            if i_level != 0:
                self.upsamp_layers.append(
                    UpsampleLayer(prev_ch)
                )
            else:
                self.upsamp_layers.append(
                    IdentityWithEmbedding()
                )

        self.decoder_layers = nn.ModuleList(self.decoder_layers)
        self.upsamp_layers = nn.ModuleList(self.upsamp_layers) 

        # Output layers
        self.out_norm = NormalizeLayer(ch_mult[0])
        self.out_conv = Conv2dSame(ch_mult[0], out_ch)

        self.cat_cond = lambda x, cond: x
        self.exp_cond = lambda x, cond: cond

        self.has_cond = (
            cond_features is not None and
            cond_emb_dim  is not None        
        )
        if self.has_cond:
            print("Conditionning !")
            self.cond_model = cond_model
            self.cond_norm  = nn.LayerNorm((cond_features))
            self.cond_layer = NIN(cond_features, cond_emb_dim)
            self.cond_activ = nonlinearity
            
            num_channels += cond_emb_dim
            
            self.exp_cond = lambda x, cond: cond.expand(x.shape[0], cond.shape[1], x.shape[2], x.shape[3])
            self.cat_cond = lambda x, cond: torch.cat((x, self.exp_cond(x, cond)), 1)

        # Initial convolution
        self.conv1 = Conv2dSame(num_channels, ch_mult[0])
    
        
    def forward(self, x, t, cond=None):
        # t = torch.randint(0, 1000, (x.shape[0], ))
        temb = get_timestep_embedding(t, self.ch * 4)
        temb = self.dense1(temb)
        temb = self.dense2(temb)

        if self.has_cond:
            cond = self.cond_model(cond)
            cond = self.cond_norm(cond)
            cond = self.cond_layer(cond)
            cond = self.cond_activ(cond)
            cond = cond.view((*cond.shape, 1, 1))

        x = self.cat_cond(x, cond)
        x = self.conv1(x)

        encoders = []
        for enc_layer, dns in zip(self.encoder_layers, self.downsamp_layers):
            current_enc = []
            for i, layer in enumerate(enc_layer):
                x = layer(self.cat_cond(x, cond), temb, self.exp_cond(x, cond))
                current_enc.append(x)

            encoders.append(current_enc[::-1])
            x = dns(x, temb, self.exp_cond(x, cond))
        
        x = self.middle(x, temb, self.exp_cond(x, cond))
        
        for layer, ups, tensors in zip(self.decoder_layers, self.upsamp_layers, reversed(encoders)):
            for sub_layers, enc in zip(layer, tensors):
                x = self.cat_cond(x, cond)
                x = sub_layers(torch.cat((x, enc), 1), temb, self.exp_cond(x, cond))

            x = ups(x, temb, self.exp_cond(x, cond))

        x = nonlinearity(self.out_norm(x))
        x = self.out_conv(x)

        return x
