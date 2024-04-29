import math
from typing import Tuple, Union, Dict

import torch
import torch.nn as nn

import utilities as utils
from utilities import AttrDict


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal Position Embeddings."""
    def __init__(self,
                 total_time_steps: int = 1000,
                 time_embed_dim: int = 128,
                 time_embed_dim_exp: int = 512
    ) -> None:
        super().__init__()

        half_dim = time_embed_dim // 2

        embed = math.log(10000) / (half_dim - 1)
        embed = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -embed)

        ts = torch.arange(total_time_steps, dtype=torch.float32)

        embed = torch.unsqueeze(ts, dim=-1) * torch.unsqueeze(embed, dim=0)
        embed = torch.cat((embed.sin(), embed.cos()), dim=-1)

        self.time_blocks = nn.Sequential(
            nn.Embedding.from_pretrained(embed),
            nn.Linear(in_features=time_embed_dim, out_features=time_embed_dim_exp),
            nn.SiLU(),
            nn.Linear(in_features=time_embed_dim_exp, out_features=time_embed_dim_exp)
        )

    def forward(self, time):
        return self.time_blocks(time)


class AttentionBlock(nn.Module):
    """Attention Block"""
    def __init__(self,
                 channels=64
    ) -> None:
        super().__init__()
        self.channels = channels

        self.group_norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.mhsa = nn.MultiheadAttention(embed_dim=self.channels, num_heads=4, batch_first=True)

    def forward(self, x):
        B, _, H, W = x.shape
        h = self.group_norm(x)
        h = h.reshape(B, self.channels, H * W).swapaxes(1, 2) # [B, C, H, W] --> [B, C, H * W] -->
        h, _ = self.mhsa(h, h, h) # [B, H*W, C]
        h = h.swapaxes(2, 1).view(B, self.channels, H, W) # [B, C, H * W] ---> [B, C, H, W]
        return x + h


class ResnetBlock(nn.Module):
    def __init__(self,
                 *,
                 in_channels,
                 out_channels,
                 dropout_rate=0.1,
                 time_embed_dims: int = 512,
                 apply_attention: bool = False
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.act_fn = nn.SiLU()
        # group 1
        self.normalize_1 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding='same')

        # group 2 time embedding
        self.dense_1 = nn.Linear(time_embed_dims, out_channels)

        # group 3
        self.normalize_2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding='same')

        if in_channels != out_channels:
            self.match_input = nn.Conv2d(in_channels, out_channels, 1, stride=1)
        else:
            self.match_input = nn.Identity()

        if apply_attention:
            self.attention = AttentionBlock(channels=out_channels)
        else:
            self.attention = nn.Identity()

    def forward(self, x, t):
        # group 1
        h = self.act_fn(self.normalize_1(x))
        h = self.conv_1(h)

        # group 2
        # add timestep embedding
        h += self.dense_1(self.act_fn(t))[:, :, None, None] # equal to two unsqueeze at the end

        # group 3
        h = self.act_fn(self.normalize_2(h))
        h = self.dropout(h)
        h = self.conv_2(h)

        # residual and attention
        h = h + self.match_input(x)
        h = self.attention(h)

        return h


class DownSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.downsample = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, *args):
        return self.downsample(x)


class UpSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, *args):
        return self.upsample(x)


class UNet(nn.Module):
    def __init__(self,
                 input_channels: int=3,
                 output_channels: int=3,
                 num_res_blocks: int=2,
                 base_channels: int=128,
                 base_channels_multiplier: Tuple[int, int, int, int] = (1, 2, 4, 8),
                 apply_attention: Tuple[bool, bool, bool, bool] = (False, False, True, False),
                 dropout_rate: float = 0.1,
                 time_multiple: int = 4
    ) -> None:
        super().__init__()

        time_embeds_dims_exp = base_channels * time_multiple
        self.time_embeddings = SinusoidalPositionEmbeddings(time_embed_dim=base_channels, time_embed_dim_exp=time_embeds_dims_exp)

        self.first = nn.Conv2d(input_channels, base_channels, 3, stride=1, padding='same')
        num_resolutions = len(base_channels_multiplier)

        # Encoder part of UNet
        self.encoder_blocks = nn.ModuleList()
        curr_channels = [base_channels]
        in_channels = base_channels

        for level in range(num_resolutions):
            out_channels = base_channels * base_channels_multiplier[level]
            for _ in range(num_res_blocks):
                block = ResnetBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout_rate=dropout_rate,
                    time_embed_dims=time_embeds_dims_exp,
                    apply_attention=apply_attention[level],
                )
                self.encoder_blocks.append(block)

                in_channels = out_channels
                curr_channels.append(in_channels)

            if level != (num_resolutions - 1):
                self.encoder_blocks.append(DownSample(channels=in_channels))
                curr_channels.append(in_channels)

        # Bottleneck in between
        self.bottleneck_blocks = nn.ModuleList(
            (
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout_rate=dropout_rate,
                    time_embed_dims=time_embeds_dims_exp,
                    apply_attention=True
                ),
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout_rate=dropout_rate,
                    time_embed_dims=time_embeds_dims_exp,
                    apply_attention=False,
                ),
            )
        )

        # Decoder part
        self.decoder_blocks = nn.ModuleList()

        for level in reversed(range(num_resolutions)):
            out_channels = base_channels * base_channels_multiplier[level]

            for _ in range(num_res_blocks + 1):
                encoder_in_channels = curr_channels.pop()
                block = ResnetBlock(
                    in_channels=encoder_in_channels + in_channels,
                    out_channels=out_channels,
                    dropout_rate=dropout_rate,
                    time_embed_dims=time_embeds_dims_exp,
                    apply_attention=apply_attention[level]
                )

                in_channels = out_channels
                self.decoder_blocks.append(block)

            if level != 0:
                self.decoder_blocks.append(UpSample(in_channels))

        self.final = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, output_channels, 3, stride=1, padding='same'),

        )

    def forward(self, x, t):
        time_embed = self.time_embeddings(t)

        h = self.first(x)
        outs = [h]

        for layer in self.encoder_blocks:
            h = layer(h, time_embed)
            outs.append(h)

        for layer in self.bottleneck_blocks:
            h = layer(h, time_embed)

        for layer in self.decoder_blocks:
            if isinstance(layer, ResnetBlock):
                out = outs.pop()
                h = torch.cat([h, out], dim=1)
            h = layer(h, time_embed)

        h = self.final(h)

        return h
    


def get_unet(configs: Union[Dict, AttrDict], 
             wandb_logger: utils.WeightAndBiases = None
) -> UNet:
    pretrained_checkpoint = None
    if configs.ModelConfig.PRETRAIN_PATH is not None:
        pretrained_checkpoint = torch.load(configs.ModelConfig.PRETRAIN_PATH) 
    model = UNet(
        input_channels=configs.ModelConfig.AUTO_EMB_CHANNELS,
        output_channels=configs.ModelConfig.AUTO_EMB_CHANNELS,
        base_channels=configs.ModelConfig.BASE_CH,
        base_channels_multiplier=configs.ModelConfig.BASE_CH_MULT,
        apply_attention=configs.ModelConfig.APPLY_ATTENTION,
        dropout_rate=configs.ModelConfig.DROPOUT_RATE,
        time_multiple=configs.ModelConfig.TIME_EMB_MULT,
    ).to(configs.BasicConfig.DEVICE)
    if pretrained_checkpoint is not None and 'unet' in pretrained_checkpoint.keys():
        print('Loading Pretrained UNet')
        model.load_state_dict(
            pretrained_checkpoint['unet'], 
            strict=True
        )
    if wandb_logger is not None:
        wandb_logger.watch(model)
    
    return model