import copy
from typing import Tuple, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import utilities as utils
from initialization import init_weights
from unet import UpSample, DownSample
from utilities import AttrDict


def swish(x: torch.Tensor):
    return x * torch.sigmoid(x)


class Swish(nn.Module):
    def __init__(self
    ) -> None:
        super().__init__()
        
    def forward(self, x):
        return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)

        self.proj_out = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** 0.5

    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        b, c, h, w = q.shape
        q = q.view(b, c, h * w)
        k = k.view(b, c, h * w)
        v = v.view(b, c, h * w)

        attn = torch.einsum('bci,bcj->bij', q, k) / self.scale
        attn = F.softmax(attn, dim=2)

        out = torch.einsum('bij,bcj->bci', attn, v)
        out = out.view(b, c, h, w)
        out = self.proj_out(out)
        return x + out


class ResnetBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(
            num_groups=8,
            num_channels=in_channels,
            eps=1e-6
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1,
                               padding=1)
        self.norm2 = nn.GroupNorm(
            num_groups=8,
            num_channels=out_channels,
            eps=1e-6
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1,
                               padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=1,
                                  padding=0)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self,
                x: torch.Tensor
    ) -> torch.Tensor:
        h = x

        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        return self.nin_shortcut(x) + h


class DualUNet(nn.Module):
    def __init__(self,
                 input_channels: int=3,
                 output_channels: int=3,
                 num_res_blocks: int=2,
                 base_channels: int=128,
                 base_channels_multiplier: Tuple[int, int, int, int] = (1, 2, 4, 8),
                 z_channels: int=1024,
                 dropout_rate: float = 0.1
    ) -> None:
        super().__init__()

        self.first = nn.Conv2d(input_channels, base_channels, 3, stride=1, padding='same')
        num_resolutions = len(base_channels_multiplier)

        # Encoder part of UNet
        self.encoder = nn.ModuleList()
        curr_channels = [base_channels]
        in_channels = base_channels

        for level in range(num_resolutions):
            out_channels = base_channels * base_channels_multiplier[level]
            for _ in range(num_res_blocks):
                block = ResnetBlock(
                    in_channels=in_channels,
                    out_channels=out_channels
                )
                self.encoder.append(block)

                in_channels = out_channels
                curr_channels.append(in_channels)

            if level != (num_resolutions - 1):
                self.encoder.append(DownSample(channels=in_channels))
                curr_channels.append(in_channels)

        # Bottleneck in between
        self.bottleneck = nn.ModuleList(
            (
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=z_channels
                ),
                ResnetBlock(
                    in_channels=z_channels,
                    out_channels=in_channels
                ),
            )
        )

        # RecDecoder part
        seg_curr_channels = copy.copy(curr_channels)
        seg_in_channels = in_channels
        
        self.rec_decoder = nn.ModuleList()
        for level in reversed(range(num_resolutions)):
            out_channels = base_channels * base_channels_multiplier[level]

            for _ in range(num_res_blocks + 1):
                encoder_in_channels = curr_channels.pop()
                block = ResnetBlock(
                    in_channels=encoder_in_channels + in_channels,
                    out_channels=out_channels
                )

                in_channels = out_channels
                self.rec_decoder.append(block)

            if level != 0:
                self.rec_decoder.append(UpSample(in_channels))

        self.rec_final = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, output_channels, 3, stride=1, padding='same')
        )
        
        # Decoder part
        self.seg_decoder = nn.ModuleList()
        for level in reversed(range(num_resolutions)):
            out_channels = base_channels * base_channels_multiplier[level]

            for _ in range(num_res_blocks + 1):
                encoder_in_channels = seg_curr_channels.pop()
                block = ResnetBlock(
                    in_channels=encoder_in_channels + seg_in_channels,
                    out_channels=out_channels
                )

                seg_in_channels = out_channels
                self.seg_decoder.append(block)

            if level != 0:
                self.seg_decoder.append(UpSample(seg_in_channels))

        self.seg_final = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=seg_in_channels),
            nn.SiLU(),
            nn.Conv2d(seg_in_channels, 1, 3, stride=1, padding='same')
        )

    def forward(self, 
                rec_data: torch.Tensor, 
                seg_data: torch.Tensor, 
                prediction=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rec_out = None
        if prediction == False: 
            # Reconstruction. 
            r = self.first(rec_data)
            outs = [r]
            for layer in self.encoder:
                r = layer(r)
                outs.append(r)
            for layer in self.bottleneck:
                r = layer(r)
            for layer in self.rec_decoder:
                if isinstance(layer, ResnetBlock):
                    out = outs.pop()
                    r = torch.cat([r, out], dim=1)
                r = layer(r)
            rec_out = self.rec_final(r)
        
        # Segmentation
        s = self.first(seg_data)
        outs = [s]
        for layer in self.encoder:
            s = layer(s)
            outs.append(s)
        for layer in self.bottleneck:
            s = layer(s)
        for layer in self.seg_decoder:
            if isinstance(layer, ResnetBlock):
                out = outs.pop()
                s = torch.cat([s, out], dim=1)
            s = layer(s)

        seg_out = self.seg_final(s)

        return rec_out, seg_out
    
    @torch.no_grad()
    def embed(self, 
              data: torch.Tensor
    ) -> torch.Tensor:
        h = self.first(data)
        for layer in self.encoder:
            h = layer(h)
        return h
    
    @torch.no_grad()
    def bottle(self, 
               data: torch.Tensor
    ) -> torch.Tensor:
        h = self.first(data)
        for layer in self.encoder:
            h = layer(h)
        for layer in self.bottleneck:
            h = layer(h)
        return h
  
 
def get_unet(configs: Union[Dict, AttrDict], 
             wandb_logger: utils.WeightAndBiases = None
) -> nn.Module:
    pretrained_checkpoint = None
    if configs.ModelConfig.PRETRAIN_PATH is not None:
        print('Loading Pretrained Model')
        pretrained_checkpoint = torch.load(configs.ModelConfig.PRETRAIN_PATH)
        
    model = DualUNet(
        input_channels=configs.ModelConfig.IN_CHANNELS,
        output_channels=configs.ModelConfig.OUT_CHANNELS,
        num_res_blocks=configs.ModelConfig.N_RESNET_BLOCKS,
        base_channels=configs.ModelConfig.CHANNELS,
        base_channels_multiplier=configs.ModelConfig.CHANNEL_MULTIPLIERS,
        z_channels=configs.ModelConfig.Z_CHANNELS,
        dropout_rate=configs.ModelConfig.DROPOUT_RATE
    )

    if (configs.ModelConfig.PRETRAIN_PATH is None and 
        configs.ModelConfig.INIT_MODEL is True
    ):  
        model.apply(
            init_weights(
                **configs.ModelConfig.INIT_MODEL_PARAMS
            )   
        )

    model = nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
    model = model.to(configs.BasicConfig.DEVICE)
    if pretrained_checkpoint is not None: 
        print('Loading Pretrained Model')
        model.load_state_dict(
            pretrained_checkpoint['model'], 
            strict=True
        )

    # Watch and log the autoencoder.
    if wandb_logger is not None:
        wandb_logger.watch(model)

    return model