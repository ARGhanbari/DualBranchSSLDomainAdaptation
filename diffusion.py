from typing import List, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as TF
from torchvision.utils import make_grid, draw_segmentation_masks
from tqdm import tqdm

import utilities as utils


class SimpleDiffusion:
    """Diffusion sampling class.
    Args: 
        num_diffusion_timesteps (int): Number of diffusion timesteps, default to 1000.
        img_shape (Tuple[int, int, int]): Image shape, default to (3, 64, 64).
        device (str): Device to use, default to 'cpu'.  
    """
    def __init__(self,
                 num_diffusion_timesteps: int = 1000,
                 img_shape: Tuple[int, int, int] = (3, 64, 64),
                 device: str = 'cpu'
    ):
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.img_shape = img_shape
        self.device = device

        self.inititialize()

    def inititialize(self):
        self.beta = self.get_betas()
        self.alpha = 1 - self.beta

        self.sqrt_beta = torch.sqrt(self.beta)
        self.alpha_cumulative = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_cumulative = torch.sqrt(self.alpha_cumulative)
        self.one_by_sqrt_alhpa = 1.0 / torch.sqrt(self.alpha)
        self.sqrt_one_minus_alpha_cumulative = torch.sqrt(1 - self.alpha_cumulative)

    def get_betas(self
    ) -> torch.Tensor:
        scale = 1000 / self.num_diffusion_timesteps
        beta_start = scale * 1e-4
        beta_end = scale * 0.02
        return torch.linspace(
            beta_start,
            beta_end,
            self.num_diffusion_timesteps,
            dtype=torch.float32,
            device=self.device
        )

    def forward_diffusion(self, 
                          x0: torch.Tensor, 
                          timesteps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        eps = torch.randn_like(x0) # noise
        mean = utils.get_tensor_element(
            self.sqrt_alpha_cumulative, 
            t=timesteps
        ) * x0 # Image
        # scaled
        std_dev = utils.get_tensor_element(
            self.sqrt_one_minus_alpha_cumulative,
            t=timesteps
        ) # Noise scaled
        sample = mean + std_dev * eps # scaled image * scaled noise

        return sample, eps



@torch.no_grad()
def reverse_diffusion(model,
                      decoder,
                      diffusion_sampler,
                      timesteps=1000,
                      img_shape=(3, 64, 64),
                      num_images=5,
                      nrow=8,
                      device='cpu',
                      inverse_mean: Union[Tuple, List, np.ndarray] = [0.0, 0.0, 0.0], 
                      inverse_std: Union[Tuple, List, np.ndarray] = [0.5, 0.5, 0.5],
                      logger: utils.WeightAndBiases=None,
                      **kwargs
) -> torch.Tensor:
    model.eval()
    decoder.eval()
    
    x = torch.randn((num_images, *img_shape), device=device)

    if kwargs.get('generate_video', False):
        outs = []

    for time_step in tqdm(iterable=reversed(range(1, timesteps)),
                          total=timesteps-1, dynamic_ncols=False,
                          desc='Sampling :: ', position=0
    ):
        ts = torch.ones(num_images, dtype=torch.long, device=device) * time_step
        z = torch.randn_like(x) if time_step > 1 else torch.zeros_like(x)

        predicted_noise = model(x, ts)

        beta_t = utils.get_tensor_element(
            diffusion_sampler.beta, ts
        )
        one_by_sqrt_alpha_t = utils.get_tensor_element(
            diffusion_sampler.one_by_sqrt_alhpa, ts
        )
        sqrt_one_minus_alpha_cumulative_t = utils.get_tensor_element(
            diffusion_sampler.sqrt_one_minus_alpha_cumulative, ts
        )

        x = (
            one_by_sqrt_alpha_t
            * (x - (beta_t / sqrt_one_minus_alpha_cumulative_t) * predicted_noise)
            + torch.sqrt(beta_t) * z
        )
    
    imgs, msks = decoder(x)
    imgs = utils.inverse_normalize(
        imgs.cpu(), 
        inverse_mean, 
        inverse_std
    ).type(torch.uint8)
    msks = torch.ge(torch.sigmoid(msks), 0.5).type(torch.bool)
    # SAVE IMAGES IN IMAGE SPACE  using wandb #
    if logger is not None:
        logger.log_image(
            images=imgs.clone(), 
            predictions=msks.cpu().clone(), 
            ground_truth=None
        )
    # SAVE IMAGES IN IMAGE SPACE #
    imgs = [
        draw_segmentation_masks(image=img, masks=msk, alpha=0.3, colors=(193,92,165))
        for img, msk in zip(imgs, msks)
    ]
    grid = make_grid(imgs, nrow=nrow, pad_value=255.0).to('cpu')
    pil_image = TF.functional.to_pil_image(grid)
    pil_image.save(kwargs['save_path'], format=kwargs['save_path'][-3:].upper())
 