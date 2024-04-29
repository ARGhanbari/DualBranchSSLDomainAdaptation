from typing import Dict, Union

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .utils import AttrDict


def get_transformations(confs: Union[AttrDict, Dict],
) -> Dict: 
    transformations = {
        'REAL_TRAIN': A.Compose(
            transforms=[
                A.Rotate(limit=[-180, 180], interpolation=1, 
                         border_mode=0, value=0, mask_value=0,
                         always_apply=False, p=0.7
                ),
                A.CenterCrop(768, 768, always_apply=True, p=1.0),
                A.HorizontalFlip(p=0.7),
                A.VerticalFlip(p=0.7),
                A.OneOf([
                    A.ElasticTransform(alpha=1, sigma=20, alpha_affine=20,
                                       interpolation=1, border_mode=0, value=0,
                                       mask_value=None, approximate=False, p=0.5),
                    A.GridDistortion(num_steps=5, distort_limit=0.3,
                                     interpolation=1,
                                     border_mode=0, value=0, mask_value=None,
                                     always_apply=False, p=0.5),
                    A.Emboss(alpha=(0.2, 0.5),
                             strength=(0.2, 0.7),
                             always_apply=False,
                             p=1.0)
                ], p=0.3),
                A.OneOf([
                    A.ToGray(p=1.0),
                    A.ToSepia(p=1.0),
                    A.FancyPCA(alpha=0.2, p=1.0),
                    A.Posterize(num_bits=4,
                                always_apply=True, p=1.0),
                    A.Sharpen(alpha=(0.2, 0.5),
                              lightness=(0.5, 1.0),
                              always_apply=False, p=1.0)
                ], p=0.3),
                A.OneOf([
                    A.RandomGamma(gamma_limit=(80, 120), eps=None,
                                  always_apply=False,
                                  p=1.0),
                    A.Equalize(mode='cv', by_channels=True, mask=None,
                               mask_params=(),
                               p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=0.2,
                                               contrast_limit=0.1,
                                               brightness_by_max=True, p=1.0),
                    A.CLAHE(clip_limit=(1.0, 5.0), tile_grid_size=(8, 8), p=1.0)
                ], p=0.3),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.Blur(blur_limit=(3, 7), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=(3, 7), p=1.0),
                    A.MedianBlur(blur_limit=(3, 7), always_apply=False, p=1.0)
                ], p=0.3),
                A.OneOf([
                    A.RandomCrop(height=768,
                                 width=768,
                                 always_apply=True
                    ),
                    A.RandomCrop(height=704,
                                 width=704,
                                 always_apply=True
                    ),
                    A.RandomCrop(height=640,
                                 width=640,
                                 always_apply=True
                    ),
                    A.RandomCrop(height=576,
                                 width=576,
                                 always_apply=True
                    ),
                    A.RandomCrop(height=512,
                                 width=512,
                                 always_apply=True
                    ),
                    A.RandomCrop(height=448,
                                 width=448,
                                 always_apply=True
                    ),
                    A.RandomCrop(height=384,
                                 width=384,
                                 always_apply=True
                    ),
                    A.RandomCrop(height=320,
                                 width=320,
                                 always_apply=True
                    ),
                    A.RandomCrop(height=256,
                                 width=256,
                                 always_apply=True
                    ),
                ], p=0.9),
                A.Resize(height=confs.IMG_SHAPE[1],
                         width=confs.IMG_SHAPE[2],
                         always_apply=True
                ),
                A.Normalize(mean=confs.TRANSFORM_MEAN,
                            std=confs.TRANSFORM_STD,
                            always_apply=True
                ),
                ToTensorV2(always_apply=True)
            ], p=1.0
        ),
        'SIMU_TRAIN': A.Compose(
            transforms=[
                A.Rotate(limit=[-180, 180], interpolation=1, 
                         border_mode=0, value=0, mask_value=0,
                         always_apply=False, p=0.7
                ),
                A.CenterCrop(768, 768, always_apply=True, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ColorJitter(brightness=0.3,
                              contrast=0.5,
                              saturation=0.5,
                              hue=0.2,
                              always_apply=False, p=0.5
                ),
                A.OneOf([
                    A.ElasticTransform(alpha=1, sigma=20, alpha_affine=20,
                                       interpolation=1, border_mode=0, value=0,
                                       mask_value=None, approximate=False, p=0.5),
                    A.GridDistortion(num_steps=5, distort_limit=0.3,
                                     interpolation=1,
                                     border_mode=0, value=0, mask_value=None,
                                     always_apply=False, p=0.5),
                    A.Emboss(alpha=(0.2, 0.5),
                             strength=(0.2, 0.7),
                             always_apply=False,
                             p=1.0)
                ], p=0.3),
                A.OneOf([
                    A.ChannelShuffle(p=1.0),
                    A.RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=50,
                               p=1.0),
                    A.ChannelDropout(channel_drop_range=(1, 2),
                                     fill_value=0,
                                     p=1.0),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30,
                                         val_shift_limit=20, p=1.0)
                ], p=0.3),
                A.OneOf([
                    A.ToGray(p=1.0),
                    A.ToSepia(p=1.0),
                    A.FancyPCA(alpha=0.2, p=1.0),
                    A.Posterize(num_bits=4,
                                always_apply=True, p=1.0),
                    A.Sharpen(alpha=(0.2, 0.5),
                              lightness=(0.5, 1.0),
                              always_apply=False, p=1.0)
                ], p=0.4),
                A.OneOf([
                    A.RandomGamma(gamma_limit=(80, 120), eps=None,
                                  always_apply=False,
                                  p=1.0),
                    A.Equalize(mode='cv', by_channels=True, mask=None,
                               mask_params=(),
                               p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=0.2,
                                               contrast_limit=0.1,
                                               brightness_by_max=True, p=1.0),
                    A.CLAHE(clip_limit=(1.0, 5.0), tile_grid_size=(8, 8), p=1.0)
                ], p=0.4),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.Blur(blur_limit=(3, 7), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=(3, 7), p=1.0),
                    A.MedianBlur(blur_limit=(3, 7), always_apply=False, p=1.0)
                ], p=0.3),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), mean=50, p=1.0),
                    A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=False,
                                        elementwise=False, p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.09), intensity=(0.1, 0.5),
                                p=1.0),
                ], p=0.5),
                A.OneOf([
                    A.RandomCrop(height=768,
                                 width=768,
                                 always_apply=True
                    ),
                    A.RandomCrop(height=704,
                                 width=704,
                                 always_apply=True
                    ),
                    A.RandomCrop(height=640,
                                 width=640,
                                 always_apply=True
                    ),
                    A.RandomCrop(height=576,
                                 width=576,
                                 always_apply=True
                    ),
                    A.RandomCrop(height=512,
                                 width=512,
                                 always_apply=True
                    ),
                    A.RandomCrop(height=448,
                                 width=448,
                                 always_apply=True
                    ),
                    A.RandomCrop(height=384,
                                 width=384,
                                 always_apply=True
                    ),
                    A.RandomCrop(height=320,
                                 width=320,
                                 always_apply=True
                    ),
                    A.RandomCrop(height=256,
                                 width=256,
                                 always_apply=True
                    ),
                ], p=0.95),
                A.Resize(height=confs.IMG_SHAPE[1],
                         width=confs.IMG_SHAPE[2],
                         always_apply=True
                ),
                A.Normalize(mean=confs.TRANSFORM_MEAN,
                            std=confs.TRANSFORM_STD,
                            always_apply=True
                ),
                ToTensorV2(always_apply=True)
            ], p=1.0
        ),
        'REAL_VALID': A.Compose(
            transforms=[
                A.CenterCrop(512, 512, always_apply=True, p=1.0),
                A.Resize(height=confs.IMG_SHAPE[1],
                         width=confs.IMG_SHAPE[2], 
                         always_apply=True
                ),
                A.Normalize(mean=confs.TRANSFORM_MEAN, 
                            std=confs.TRANSFORM_STD,
                            always_apply=True
                ),
                ToTensorV2(always_apply=True)
            ], p=1.0
        ),
        'SIMU_VALID': A.Compose(
            transforms=[
                A.CenterCrop(512, 512, always_apply=True, p=1.0),
                A.Resize(height=confs.IMG_SHAPE[1],
                         width=confs.IMG_SHAPE[2],
                         always_apply=True
                         ),
                A.Normalize(mean=confs.TRANSFORM_MEAN,
                            std=confs.TRANSFORM_STD,
                            always_apply=True
                            ),
                ToTensorV2(always_apply=True)
            ], p=1.0
        ),
        'TEST': A.Compose(
            transforms=[
                A.CenterCrop(512, 512, always_apply=True, p=1.0),
                A.Resize(height=confs.IMG_SHAPE[1],
                         width=confs.IMG_SHAPE[2], 
                         always_apply=True
                ),
                A.Normalize(mean=confs.TRANSFORM_MEAN, 
                            std=confs.TRANSFORM_STD,
                            always_apply=True
                ),
                ToTensorV2(always_apply=True)
            ], p=1.0
        ),
        'PRED': A.Compose(
            transforms=[
                A.CenterCrop(512, 512, always_apply=True, p=1.0),
                A.Resize(height=confs.IMG_SHAPE[1],
                         width=confs.IMG_SHAPE[2], 
                         always_apply=True
                ),
                A.Normalize(mean=confs.TRANSFORM_MEAN, 
                            std=confs.TRANSFORM_STD,
                            always_apply=True
                ),
                ToTensorV2(always_apply=True)
            ], p=1.0
        )
    }
    return transformations
    