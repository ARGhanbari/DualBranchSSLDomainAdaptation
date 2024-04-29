import copy
import random
from typing import Callable, List, Dict, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

import utilities as utils
from utilities import AttrDict


class ImageDataset(torch.utils.data.Dataset):
    """Wheat Head Segmentation data loader.

    Args:
        metadata_paths (sequence): A list of metadata paths. Each path must
            be in csv format.
        as_tensor (boolean): Return image and its contour as a tensor or as a
            numpy array.
        transformer (Callable): A composition of a list of augmentation
            transformations.
    Returns:
        image (pytorch FloatTensor): A new generated image by generator object.
        mask (pytorch UIntTensor): A new generated contour image by generator
            object for the image.
    """
    def __init__(self,
                 metadata: utils.Metadata,
                 subset_size: int = -1,
                 transform: Union[Callable, None]=None,
                 columns: Union[List, None]=None,
                 min_intensity: int = -400,
                 max_intensity: int = 400):
        super(ImageDataset, self).__init__()
        self.transform = transform
        if columns is None:
            self.columns = copy.deepcopy(metadata.columns)
        else:
            self.columns = copy.deepcopy(columns)

        self.metadata = metadata
        self.subset_size = subset_size
        self.subset_sampler()

        self.image_loader = utils.ImageLoader(
            out_dtype=np.uint8,
            clip=False,
            scaler=False
        )
        self.mask_loader = utils.MaskLoader(
            out_dtype=np.uint8,
            binary=True
        )
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity

    def subset_sampler(self): 
        if 0 < self.subset_size < len(self.metadata):
            print(f"Apply subset sampler with size of {self.subset_size}/{len(self.metadata)}.")
            self.metadata.set_to_subset(self.subset_size)

    def __len__(self):
        return (
            self.subset_size 
            if 0 < self.subset_size < len(self.metadata) 
            else len(self.metadata)
        )

    def __getitem__(self, item: int):
        sample = self.metadata[item]
        image = sample.get('Image', None)
        mask = sample.get('Mask', None)
        datapoint = {}
        for c in self.columns:
            if c == 'Image':
                datapoint[c] = self.image_loader(image)
            elif c == 'Mask':
                msk = None
                if mask is not None:
                    msk = self.mask_loader(mask)
                datapoint[c] = msk
            else:
                datapoint[c] = sample[c]
        if self.transform is not None:
            if datapoint.get('Mask', None) is not None:
                augmented = self.transform(
                    image=datapoint['Image'], mask=datapoint['Mask']
                )
                datapoint['Image'] = augmented['image']
                datapoint['Mask'] = augmented['mask']
                if datapoint['Mask'].ndim == 2:
                    datapoint['Mask'] = datapoint['Mask'].unsqueeze(dim=0).long()
            else:
                datapoint['Image'] = self.transform(image=datapoint['Image'])['image']
              
        if datapoint['Mask'] is None: 
            datapoint['Mask'] = torch.zeros(datapoint['Image'].shape[1:]).type(torch.uint8)
        
        return (sample.get('GID', None), sample.get('IID', None),
                datapoint['Image'], datapoint['Mask']
        )

def get_dataset(metadata_paths: List,
                subset_size: int,
                transforms: Callable
) -> ImageDataset:
    metadata = utils.Metadata(metadata_paths=metadata_paths)
    dataset = ImageDataset(
        metadata=metadata,
        subset_size=subset_size,
        transform=transforms,
        columns=['GID', 'IID', 'Image', 'Mask', 'Label']
    )
    return dataset

def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)    

def get_dataloader(metadata_paths: List, 
                   subset_size: int,
                   transformations: Callable, 
                   confs: Union[Dict, AttrDict]=None,
                   phase: str = 'TRAIN', 
                   seed: int = 123
) -> DataLoader:
    dataset = get_dataset(
        metadata_paths=metadata_paths,
        subset_size=subset_size,
        transforms=transformations
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    dataloader = DataLoader(
        dataset,
        batch_size=confs.BATCH_SIZE,
        pin_memory=confs.PIN_MEMORY,
        num_workers=confs.NUM_WORKERS,
        shuffle=(confs.SHUFFLE and phase.upper() == 'TRAIN'),
        worker_init_fn=seed_worker,
        generator=loader_generator
    )
    return dataloader