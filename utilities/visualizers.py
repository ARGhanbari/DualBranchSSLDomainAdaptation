import base64
from typing import Union, Tuple, List

import cv2
import numpy as np
import torch
from IPython.display import display, HTML
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid, draw_segmentation_masks


def frames2vid(images: List, 
               save_path: str
) -> None:
    """Convert a list of images to a video.

    Args:
        images (List): _description_
        save_path (str): _description_
    """
    WIDTH = images[0].shape[1]
    HEIGHT = images[0].shape[0]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path, fourcc, 25, (WIDTH, HEIGHT))

    for image in images:
        video.write(image)

    video.release()

def display_gif(gif_path):
    """Display a gif. Useful in in jupyter notebook.

    Args:
        gif_path (_type_): _description_
    """
    b64 = base64.b64encode(open(gif_path,'rb').read()).decode('ascii')
    display(HTML(f'<img src="data:image/gif;base64,{b64}" />'))

def draw_segmentation_map(images: Union[List, torch.Tensor],
                          masks: Union[List, torch.Tensor],
                          nrows: int=None,
                          colors: Union[Tuple, List]=(204,68,150),
                          out_path: str=None, 
                          visualize: bool=False,
) -> None:
    """Draw overlayed segmentation map on images.
    Args:
        images (List, torch.Tensor): images to draw segmentation map on.
        masks (List, torch.Tensor, optional): masks to overlay on the images. Defaults to None.
        nrows (_type_, optional): number of rows for the grid. Defaults to None.
        colors (tuple, optional): colors for the segmnetation maps. Defaults to (193,92,165).
        out_path (_type_, optional): path to save overlaid images. Defaults to None.
        visualize (bool, optional): whether visualize the map or not. Defaults to True.
    Returns:
        None
    """
    assert isinstance(images, (List, torch.Tensor)), "images must be a torch.Tensor"
    assert isinstance(masks, (List, torch.Tensor)), "masks must be a torch.Tensor"
    if nrows is None:
        nrows = int(np.sqrt(len(images)))
    images_with_masks = [
        draw_segmentation_masks(
            img.squeeze(), 
            masks=msk.type(torch.bool).squeeze(), 
            colors=colors, alpha=0.4
        )
        for img, msk in zip(images, masks)
    ]
    grid = make_grid(images_with_masks, nrow=nrows, padding=3, pad_value=0)
    grid = ToPILImage()(grid)
    if out_path is not None:
        grid.save(out_path, dpi=(600, 600))
    if visualize is True:
        grid.show()
        
def draw_grid_map(images: Union[List, torch.Tensor],
                  nrows: int=None,
                  out_path: str=None, 
                  visualize: bool=False,
) -> None:
    """Draw a grid of input images, save or visualize.
    Args:
        images (torch.Tensor): images to draw the grid map.
        nrows (_type_, optional): number of rows for the grid. Defaults to None.
        colors (tuple, optional): colors for the segmnetation maps. Defaults to (193,92,165).
        out_path (_type_, optional): path to save overlaid images. Defaults to None.
        visualize (bool, optional): whether visualize the map or not. Defaults to True.
    Returns:
        None
    """
    assert isinstance(images, (List, torch.Tensor)), "images must be a torch.Tensor"
    if nrows is None:
        nrows = int(np.sqrt(len(images)))
    grid = make_grid([img for img in images], nrow=nrows, padding=3, pad_value=0)
    grid = ToPILImage()(grid)
    if out_path is not None:
        grid.save(out_path, dpi=(600, 600))
    if visualize is True:
        grid.show()