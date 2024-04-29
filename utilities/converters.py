from typing import Union, List, Tuple

import cv2
import numpy as np
import torch


def channel_broadcast(image: np.ndarray,
                      num_channels: int
) -> np.ndarray:
    """Broadcasts image to the specified number of channels from 2d imags or 3d image with 1 channel"""
    image = image.squeeze()
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)
        image = np.concatenate(
            (image, ) * num_channels, axis=-1
        ).astype(np.uint8)
    else:
        raise ValueError('Image is not in the correct format. Expected format: 2d or 3d with 1 channel')
    return image

def inverse_normalize(data: torch.Tensor,
                      mean: Union[Tuple, List, np.ndarray, torch.Tensor],
                      std: Union[Tuple, List, np.ndarray, torch.Tensor]
) -> torch.Tensor:
    """Inverse transform data to its original form.

    Args:
        data (Union[np.ndarray, torch.Tensor]): data to inverse.
        mean (Union[Tuple, List]): the same mean values used for 
            transformaing data, now used to inverse the transform.
        std (Union[Tuple, List]): the same std values used for 
            transformaing data, now used to inverse the transform.
    Returns:
        torch.Tensor: inversed data would be the same size and shape as the input.
    """
    assert isinstance(data, torch.Tensor), "data must be a torch.Tensor"
    mean = torch.tensor(mean).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)
    std = torch.tensor(std).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)
    
    data = (data * std) + mean
    data = (data - data.min()) / (data.max() - data.min())
    data = (data * 255.0).type(torch.uint8)
    
    return data

def segmap2rgbmap(mask: Union[torch.Tensor]
) -> np.ndarray:
    """Decode segmentation map to an RGB map

    Args:
        mask: (numpy.array) the segmentation image
        nc: (int) number of classes that segmentation have
    Returns: 
        RGB image of type Tensor. 
    """
    mask = mask.detach().cpu().squeeze()
    nc = torch.unique(mask).tolist()
    if len(mask.shape) == 3:
        mask = torch.argmax(mask, axis=0)

    label_colors = torch.tensor(
        [
            (0, 0, 0),  # 0=Unlabeled
            (193,92,165), # Wheat Head
        ]
    )

    r = torch.zeros_like(mask).type(torch.uint8)
    g = torch.zeros_like(mask).type(torch.uint8)
    b = torch.zeros_like(mask).type(torch.uint8)

    for i, l in enumerate(nc):
        idx = mask == l
        r[idx] = label_colors[i, 0]
        g[idx] = label_colors[i, 1]
        b[idx] = label_colors[i, 2]

    rgb = torch.stack([r, g, b], axis=0)
    return rgb


class ColorSpaceConverter(object):
    """Converts image color space to another format
    Args:
        source_format (str): source image format. Supported formats are `RGB`, `BGR`, `HSV`, `HLS`, `LAB`, `GRAY`.
            Default is `RGB`.
    """
    def __init__(self,
                source_format: str = 'RGB'
    ) -> None:
        self.source_format = source_format

    def check(self,
              image: np.ndarray
    ) -> None:
        """Checks if image is in the correct format
        Args:
            image: image to check
        Returns:
            True if image is in the correct format, False otherwise
        """
        if self.source_format in ['RGB', 'BGR', 'HSV', 'LAB', 'HLS']:
            return image.ndim == 3 and image.shape[2] == 3
        elif self.source_format == 'GRAY':
            return image.ndim == 2
        else:
            raise ValueError('Unknown format: {}'.format(self.source_format))
    def tobgr(self,
              image: np.ndarray
    ) -> np.ndarray:
        """Converts image to BGR format
        Args:
            image: image to convert
        Returns:
            image in BGR format
        """
        if self.check(image):
            if self.source_format == 'RGB':
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if self.source_format == 'GRAY':
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            raise ValueError('Image is not in the correct format. Expected format: {}'.format(self.source_format))
        return image
    def torgb(self,
              image: np.ndarray
    ) -> np.ndarray:
        """Converts image to RGB format
        Args:
            image: image to convert
        Returns:
            image in RGB format
        """
        if self.check(image):
            if self.source_format == 'BGR':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.source_format == 'GRAY':
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            raise ValueError('Image is not in the correct format. Expected format: {}'.format(self.source_format))
        return image

    def togray(self,
               image: np.ndarray
    ) -> np.ndarray:
        """Converts image to GRAY format
        Args:
            image: image to convert
        Returns:
            image in GRAY format
        """
        if self.check(image):
            if self.source_format == 'RGB':
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if self.source_format == 'BGR':
                image = cv2.cvtColor(image, cv2.Color_BGR2GRAY)
        else:
            raise ValueError('Image is not in the correct format. Expected format: {}'.format(self.source_format))
        return image

    def tohsv(self,
              image: np.ndarray
    ) -> np.ndarray:
        """Converts image to HSV format
        Args:
            image: image to convert
        Returns:
            image in HSV format
        """
        if self.check(image):
            if self.source_format == 'BGR':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif self.source_format == 'RGB':
                image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            else:
                raise ValueError('Only RGB or BGR source color spaces are allowed for '
                                 'converting to HSV color space. Source format is {}'.format(self.source_format))
        else:
            raise ValueError('Image is not in the correct format. Expected format: {}'.format(self.source_format))
        return image

    def tolab(self,
              image: np.ndarray
    ) -> np.ndarray:
        """Converts image to LAB format
        Args:
            image: image to convert
        Returns:
            image in LAB format
        """
        if self.check(image):
            if format == 'BGR':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            elif format == 'RGB':
                image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            else:
                raise ValueError('Only RGB or BGR source color spaces are allowed for '
                                 'converting to LAB color space. Source format is {}'.format(self.source_format))
        else:
            raise ValueError('Image is not in the correct format. Expected format: {}'.format(self.source_format))
        return image

    def tohls(self,
              image: np.ndarray
    ) -> np.ndarray:
        """Converts image to HLS format
        Args:
            image: image to convert
        Returns:
            image in HLS format
        """
        if self.check(image):
            if self.source_format == 'BGR':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            elif self.source_format == 'RGB':
                image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            else:
                raise ValueError('Only RGB or BGR source color spaces are allowed for '
                                 'converting to LAB color space. Source format is {}'.format(self.source_format))
        else:
            raise ValueError('Image is not in the correct format. Expected format: {}'.format(self.source_format))
        return image


class IntensityScaler(object):
    """Converts image intensity to another intensity"""

    def clip_intensity(self,
                       image: np.ndarray,
                       min_intensity: Union[int, float],
                       max_intensity: Union[int, float],
    ) -> np.ndarray:
        """Clips image intensity to the specified range
        Args:
            image: image to clip
            min_intensity: minimum intensity value
            max_intensity: maximum intensity value
        Returns:
            image with clipped intensity
        """
        image = np.clip(image, min_intensity, max_intensity)
        image = image.astype(np.float32)
        return image

    def min_max_scaler(self,
                       image: np.ndarray
    ) -> np.ndarray:
        """Scale the input image to [0, 1] range
        Args:
            image: image to convert
        Returns:
            scaled image in [0, 1] range
        """
        image = (image - image.min()) / (image.max() - image.min())
        image = self.clip_intensity(image, min_intensity=0.0, max_intensity=1.0)
        return image

    def standard_scaler(self,
                        image: np.ndarray
    ) -> np.ndarray:
        """Scale the input image to mean=0 and std=1
        Args:
            image: image to convert
        Returns:
            scaled image with mean=0 and std=1
        """
        image = (image - image.mean()) / image.std()
        image = self.clip_intensity(image, min_intensity=0.0, max_intensity=1.0)
        return image

    def scale(self,
              image: np.ndarray,
              min_intensity: Union[int, float],
              max_intensity: Union[int, float]
    ) -> np.ndarray:
        """Scale the input image to the specified range
        Args:
            image: image to convert
            min_intensity: minimum intensity value
            max_intensity: maximum intensity value
        Returns:
            scaled image
        """
        image = self.min_max_scaler(image)
        image = image * (max_intensity - min_intensity) + min_intensity
        image = self.clip_intensity(image, min_intensity=min_intensity, max_intensity=max_intensity)
        return image