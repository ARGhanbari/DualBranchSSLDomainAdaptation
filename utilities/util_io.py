from typing import Dict, Optional, Callable, List, Union

import SimpleITK as sitk
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from box import Box
from skimage import io
from torch.utils.data import Sampler

from .converters import IntensityScaler
from .converters import channel_broadcast
from .utils import AttrDict

# Global configs. 
plt.rcParams["savefig.bbox"] = 'tight'


class ImageLoader(object):
    """Loads image from file
    Args:
        image_path: path to image file
        clip: whether to clip the intensity of the image
        clip_min: minimum value of the intensity
        clip_max: maximum value of the intensity
        scaler: whether to scale the intensity of the image
        scaler_func: function to scale the intensity of the image. Options: `MinMaxScaler`, `StandardScaler`, `Scale`.
            Default is 'min_max_scaler'.
        scaler_args: arguments for the scaler function, only for scale function. It includes `min_intensity` and `max_intensity`.
    """
    def __init__(self,
                 out_dtype: Optional[Callable] = 'float32',
                 clip: bool = False,
                 clip_min: Optional[float] = None,
                 clip_max: Optional[float] = None,
                 scaler: bool = False,
                 scaler_func: Optional[Callable] = 'MinMaxScaler',
                 scaler_args: Optional[Dict] = None
    ) -> None:
        self.out_dtype = out_dtype
        self.clip = clip
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.scaler = scaler
        self.scaler_func = scaler_func
        self.scaler_args = scaler_args
        if self.clip is True:
            assert self.clip_min is not None, 'clip_min must be specified!'
            assert self.clip_max is not None, 'clip_max must be specified!'
            assert self.clip_min < self.clip_max, 'clip_min must be less than clip_max!'
        if self.scaler is True and self.scaler_func == 'Scale':
            assert self.scaler_args is not None, 'scaler_args must be specified!'
            assert 'min_intensity' in self.scaler_args, 'min_intensity must be specified!'
            assert 'max_intensity' in self.scaler_args, 'max_intensity must be specified!'
            assert self.scaler_args['min_intensity'] < self.scaler_args['max_intensity'], 'min_intensity must be less than max_intensity!'
        self.intensity_scaler = IntensityScaler()

    @staticmethod
    def load_npy(path: str
    ) -> np.ndarray:
        """Load numpy files. The path must end in `.npy`.
        Args:
            path (str): Path to the numpy file.
        Returns:
            np.ndarray: The loaded numpy array.
        """
        assert path.endswith('.npy'), 'Only `npy` files are allowed!'
        image = np.load(path)
        return image

    @staticmethod
    def load_png(path: str
    ) -> np.ndarray:
        """Load an image file in png format. The path must end in `.png`.
        Args:
            path (str): Path to the numpy file.
        Returns:
            np.ndarray: The loaded numpy array.
        """
        assert path.endswith('.png'), 'Only `npy` files are allowed!'
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    @staticmethod
    def load_nrrd(path: str
    ) -> np.ndarray:
        """Load an image file in nrrd format. The path must end in `.nrrd`.
        Args:
            path (str): Path to the numpy file.
        Returns:
            np.ndarray: The loaded numpy array.
        """
        assert path.endswith('.nrrd'), 'Only `nrrd` files are allowed!'
        image = sitk.GetArrayFromImage(sitk.ReadImage(path)).squeeze()
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        return image

    def __call__(self, 
                 image_path: str
    ) -> None:
        image = None
        if image_path.endswith('.npy'):
            image = ImageLoader.load_npy(image_path)
        elif image_path.endswith('.png'):
            image = ImageLoader.load_png(image_path)
        elif image_path.endswith('.nrrd'):
            image = ImageLoader.load_nrrd(image_path)
        else:
            image = io.imread(image_path)
        # Remove Dimensions of size 1.
        if self.clip is True:
            image = self.intensity_scaler.clip_intensity(image, self.clip_min, self.clip_max)
        if self.scaler is True:
            if self.scaler_func == 'MinMaxScaler':
                image = self.intensity_scaler.min_max_scaler(image)
            elif self.scaler_func == 'StandardScaler':
                image = self.intensity_scaler.standard_scaler(image)
            elif self.scaler_func == 'Scale':
                image = self.intensity_scaler.scale(image, **self.scaler_args)
            else:
                raise ValueError('Invalid scaler function!')
        if image.ndim == 2 or image.shape[-1] == 1:
            image = channel_broadcast(image, num_channels=3)
        if self.out_dtype == 'uint8' and image.max() <= 1:
            image = image * 255
        image = image.astype(self.out_dtype)
        return image


class MaskLoader(object):
    def __init__(self,
                out_dtype: Optional[str] = np.uint8,
                binary: bool = False
    ) -> None:
        self.binary = binary
        self.out_dtype = np.uint8 if out_dtype is None else out_dtype

    def __call__(self,
                 mask_path: str
    ) -> np.ndarray:
        if mask_path.endswith('.npy'):
            mask = ImageLoder.load_npy(mask_path)
        elif mask_path.endswith('.nrrd'):
            mask = ImageLoader.load_nrrd(mask_path)
        else:
            mask = io.imread(mask_path)
        if self.binary is True:
            mask[mask > 0] = 1
        mask = mask.squeeze()
        mask = mask.astype(self.out_dtype)
        return mask


class VideoLoader(object):
    pass


class Metadata(object):
    def __init__(self, metadata_paths: List):
        frames = []
        for pth in metadata_paths:
            frame = pd.read_csv(pth)
            frames.append(frame)
        df = pd.concat(frames)
        df.reset_index(drop=True, inplace=True)
        self.columns = df.columns
        self.df = df
        self.reset()

    def __call__(self):
        return self.df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        return self.df_dict[item]
    
    def set_to_subset(self, subset_size):
        indices = np.random.randint(0, self.__len__(), subset_size)
        self.df_dict = self.df.iloc[
            indices
        ].reset_index(
            drop=True
        ).to_dict(orient='index')

    def reset(self): 
        self.df_dict = self.df.to_dict(orient='index')
  

class DynamicRandomSampler(Sampler):
    """

    Args:
        metadata: A metadata object containing the metadata for a dataset
        categories: Array-like of categories for each sample

    """
    def __init__(self, metadata: Metadata, category: str):
        self.metadata = metadata
        self.categories = np.array(copy.deepcopy(metadata.df[category]))
        unique_categories = np.unique(metadata.df.loc[:, category])
        self.category_probability = {c: 1 for c in unique_categories}
        self.sampels = np.random.permutation(len(metadata))

    def __iter__(self):
        return iter(copy.deepcopy(self.sampels))

    def set_probability(self, category_probability:dict):
        prob = np.random.uniform(low=0, high=1, size=len(self.categories))
        selected = [True if p <= category_probability[c] else False
                    for c, p in zip(self.categories, prob)]
        sampels = np.arange(len(self.metadata))
        self.sampels = sampels[selected]
        np.random.shuffle(self.sampels)

    def __len__(self) -> int:
        return len(self.sampels)


class ConfigLoader(object):
    """Load configuration either from a `YAML` or `JSON` file.
        Args:
            config_path (str): The path to the config file.
        Returns:
            configs (Dict): a dictionary contains all the configuration parameters.
    """
    def load_yaml(self,
                  config_path: str
    ) -> Union[Dict, AttrDict, Box]:
        configs = Box.from_yaml(
            filename=config_path, 
            Loader=yaml.FullLoader
        ) 
        return configs

    def load_json(self,
                  config_path: str
    ) -> Union[Dict, AttrDict, Box]:
        configs = Box.from_json(filename=config_path)
        return configs

    def __call__(self,
                 config_path: str
    ) -> Union[Dict, AttrDict, Box]:
        if config_path.endswith('.json'):
            configs = self.load_json(config_path)
        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            configs = self.load_yaml(config_path)
        else:
            raise ValueError('Only `Json` or `Yaml` configs are acceptable.')
        return configs