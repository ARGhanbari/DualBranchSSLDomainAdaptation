import os
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
from skimage import io
from torch import nn
from tqdm import tqdm

import utilities as utils
from diffusion import SimpleDiffusion


class Development: 
    def __init__(self, 
                 model: nn.Module,
                 loaders: Dict[str, torch.utils.data.DataLoader],
                 diffusion_sampler: SimpleDiffusion,
                 diffusion_patch_size: int,
                 time_steps: int,
                 optimizer: torch.optim.Optimizer,
                 img_loss_fn: nn.Module,
                 msk_loss_fn: nn.Module,
                 msk_score_fn: Callable,
                 epochs: int = 100,
                 device: int=0,
                 log_dir: str=None, 
                 logger: Callable=None,
                 inverse_mean: Union[List, Tuple]=(0.0, 0.0, 0.0),
                 inverse_std: Union[List, Tuple]=(1.0, 1.0, 1.0)
    ) -> None:
        self.model = model
        self.loaders = loaders
        self.diffusion_sampler = diffusion_sampler
        self.diffusion_patch_size = diffusion_patch_size
        self.time_steps = time_steps
        self.optimizer = optimizer
        self.img_loss_fn = img_loss_fn
        self.msk_loss_fn = msk_loss_fn
        self.msk_score_fn = msk_score_fn
        self.epochs = epochs
        self.device = device
        self.log_dir = log_dir
        self.logger = logger
        self.inverse_mean = inverse_mean
        self.inverse_std = inverse_std
        
        self.local_logger = utils.Logger(
            log_dir=log_dir if log_dir is not None else 'out/',
            log_file_name='experiment.log',
            log_mode='a', 
            add_stream_handler=False,
            stream_terminator=''
        )
    
    def __call__(self, 
                 phase: str,
                 epoch: int
    ) -> float:
        with tqdm(total=min(len(self.loaders[phase]['REAL']), 
                            len(self.loaders[phase]['SIMU'])), 
                  dynamic_ncols=True) as tq:
            mean_loss, mean_rec_loss, mean_seg_loss, mean_score = self.ende_forward(phase, epoch, tq)
            tq.set_postfix_str(
                s=f"Total Loss -> {mean_loss: 0.4f}, Metrics -> {utils.make_message(mean_score)}"
            )
            self.local_logger.log({
                f"Phase": phase,
                f"Epoch": f"{epoch}/{self.epochs}",
                f"{phase} Loss": mean_loss,
                f"{phase} Dice": mean_score['dice'],
                f"{phase} IoU": mean_score['iou']
            })
        return mean_loss, mean_rec_loss, mean_seg_loss, mean_score

    def ende_forward(self, 
                     phase: str, 
                     epoch: int, 
                     tq: tqdm
    ) -> Tuple[float, float, float, float]: 
        loss_record = utils.MeanMetric(scalar=True)
        rec_loss_record = utils.MeanMetric(scalar=True)
        seg_loss_record = utils.MeanMetric(scalar=True)
        score_record = utils.MeanMetric(scalar=False)
    
        message = f"{phase} :: Epoch: {epoch}/{self.epochs}"
        tq.set_description(message)
        for (_, _, real_images, _), (_, _, simu_images, simu_masks) in zip(
                self.loaders[phase]['REAL'], 
                self.loaders[phase]['SIMU']
        ):
            tq.update(1)
            
            real_images = real_images.to(self.device)
            simu_images = simu_images.to(self.device)
            simu_masks = simu_masks.to(self.device)
        
            ts = torch.randint(low=1, high=self.time_steps,
                            size=(real_images.shape[0],),
                            device=self.device
            )
            noisy_real_images, _ = self.diffusion_sampler.forward_diffusion(
                real_images.clone(), ts
            ) 
            # Mix real and noisy images.
            x = torch.randint(low=0, high=real_images.shape[2]-self.diffusion_patch_size, size=(1,)).item()
            y = torch.randint(low=0, high=real_images.shape[3]-self.diffusion_patch_size, size=(1,)).item()
            contour_map = torch.zeros_like(real_images)
            contour_map[:, :, x:x+self.diffusion_patch_size, y:y+self.diffusion_patch_size] = 1.0
            noisy_real_images = (
                (noisy_real_images * contour_map) + 
                (real_images * torch.abs(1.0 - contour_map))
            )
            with torch.set_grad_enabled(phase == 'TRAIN'):
                ## Run the forward pass under autocast. 
                # with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred_images, pred_masks = self.model(noisy_real_images, simu_images)
                
                img_loss = self.img_loss_fn(pred_images, real_images)
                msk_loss = self.msk_loss_fn(pred_masks, simu_masks)

                total_batch_loss = img_loss + msk_loss
                if phase == 'TRAIN':
                    self.optimizer.zero_grad()
                    total_batch_loss.backward()
                    self.optimizer.step()
                
                loss_record.update(total_batch_loss.item())
                rec_loss_record.update(img_loss.item())
                seg_loss_record.update(msk_loss.item())
                
                metric_values = self.msk_score_fn(pred_masks, simu_masks)
                score_record.update(metric_values)

                tq.set_postfix_str(
                    s=f"Total Loss --> {total_batch_loss:0.3f}, Metrics --> {utils.make_message(metric_values)}"
                )
        # Decode mask and visualization.
        real_images = real_images.detach().cpu()[:min(4, real_images.shape[0])].float()
        noisy_real_images = noisy_real_images.detach().cpu()[:min(4, noisy_real_images.shape[0])].float()
        simu_images = simu_images.detach().cpu()[:min(4, simu_images.shape[0])].float()
        simu_masks = simu_masks.detach().cpu()[:min(4, simu_masks.shape[0])].float()
        pred_images = pred_images.detach().cpu()[:min(4, pred_images.shape[0])].float()
        pred_masks = pred_masks.detach().cpu()[:min(4, pred_masks.shape[0])].float()
        real_images = utils.inverse_normalize(
            real_images,
            mean=self.inverse_mean,
            std=self.inverse_std
        )
        noisy_real_images = utils.inverse_normalize(
            noisy_real_images,
            mean=self.inverse_mean,
            std=self.inverse_std
        )
        simu_images = utils.inverse_normalize(
            simu_images,
            mean=self.inverse_mean,
            std=self.inverse_std
        )
        pred_images = utils.inverse_normalize(
            pred_images, 
            mean=self.inverse_mean,
            std=self.inverse_std
        )
        pred_masks = torch.ge(torch.sigmoid(pred_masks), 0.5).float()
        
        epoch_dir_name = f"Epoch{epoch:0>4}/{phase}/"
        os.makedirs(os.path.join(self.log_dir, epoch_dir_name), exist_ok=True)
        
        Development.epoch_visualizer(
            real_images, noisy_real_images, 
            simu_images, simu_masks,
            pred_images, pred_masks, 
            epoch, self.log_dir, epoch_dir_name, phase
        )
        if isinstance(self.logger, utils.WeightAndBiases):
            self.logger.log_image(
                images=simu_images,
                predictions=pred_masks, 
                ground_truth=simu_masks
            )                
        return loss_record.compute(), rec_loss_record.compute(), seg_loss_record.compute(), score_record.compute()

    def add_noise(self, 
                  images: torch.Tensor, 
                  time_steps: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns noisy images and masks. Mask is 1.0 where the noise is added."""
        # TODo: add this to the configs later. 
        noise_square_size = 64
        
        out_images = images.clone()
        out_masks = torch.zeros((images.shape[0], 1, images.shape[2], images.shape[3]), 
                                 dtype=torch.float32
        ).to(self.device)
        
        ts = torch.randint(low=1, high=time_steps,
                           size=(out_images.shape[0],),
                           device=self.device
        )
        noise, _ = self.diffusion_sampler.forward_diffusion(
            out_images.clone(), ts
        ) 
        
        for _ in range(4): 
            x = torch.randint(low=0, high=out_images.shape[2]-noise_square_size, size=(1,)).item()
            y = torch.randint(low=0, high=out_images.shape[3]-noise_square_size, size=(1,)).item()
            
            out_images[:, :, 
                       x:x+noise_square_size, 
                       y:y+noise_square_size] = noise[:, :, 
                                                      x:x+noise_square_size, 
                                                      y:y+noise_square_size]
            out_masks[:, :, x:x+noise_square_size, y:y+noise_square_size] = 1.0
            
        return out_images, out_masks            
    
    @staticmethod
    def predict(model: nn.Module, 
                loaders: Callable,
                device: Union[str, torch.device],
                inverse_mean: Union[Tuple, List], 
                inverse_std: Union[Tuple, List],
                prediction_dir: str
    ) -> None: 
        with tqdm(total=min(len(loaders['REAL']), len(loaders['SIMU'])), 
                  dynamic_ncols=True
        ) as tq:
            message = f"{'PRED'} :: Epoch: {0}/{0}"
            tq.set_description(message)
            image_counter = 0
            for (iids, gids, real_images, _), _ in zip(
                    loaders['REAL'], 
                    loaders['SIMU']
            ):
                tq.update(1)
                
                real_images = real_images.to(device)
                with torch.set_grad_enabled(False):
                    _, pred_masks = model(real_images.clone(), real_images, prediction=True)
                
                real_images = real_images.detach().cpu().float()
                real_images = utils.inverse_normalize(
                    real_images,
                    mean=inverse_mean,
                    std=inverse_std
                )
                real_images = real_images.numpy().transpose(0, 2, 3, 1)
                
                pred_masks = pred_masks.detach().cpu().float()
                pred_masks = torch.ge(torch.sigmoid(pred_masks), 0.5).float()
                pred_masks = (255.0 * pred_masks).numpy().astype(np.uint8)
                for gid, iid, img, msk in zip(
                    gids, iids, real_images, pred_masks
                ): 
                    io.imsave(
                        os.path.join(prediction_dir, 
                                    f"{gid}_{iid}_img_{image_counter:0>5}.png"
                        ), 
                        img, 
                        check_contrast=False
                    )
                    io.imsave(
                        os.path.join(prediction_dir, 
                                    f"{gid}_{iid}_msk_{image_counter:0>5}.png"
                        ), 
                        msk, 
                        check_contrast=False
                    )
                    image_counter += 1
    
    @staticmethod
    def epoch_visualizer(real_images: torch.Tensor,
                         noisy_images: torch.Tensor,
                         simu_images: torch.Tensor,
                         simu_masks: torch.Tensor,
                         pred_images: torch.Tensor,
                         pred_masks: torch.Tensor, 
                         epoch: int,
                         log_dir: str,
                         epoch_dir_name: str, 
                         phase: str
    ) -> None:         
        utils.draw_segmentation_map(
            images=simu_images, 
            masks=simu_masks,
            nrows=2,
            out_path=os.path.join(log_dir, epoch_dir_name,
                        f"{phase}-{epoch:0>3}-SimulatedImageMask_Groundtruth.png")
        )
        utils.draw_grid_map(
            images=[utils.segmap2rgbmap(item) 
                    for item in simu_masks
            ],             
            nrows=2,
            out_path=os.path.join(log_dir, epoch_dir_name,
                            f"{phase}-{epoch:0>3}-SimulatedMask_Groundtruth.png")
        )
        utils.draw_segmentation_map(
            images=simu_images,
            masks=pred_masks,
            nrows=2,
            out_path=os.path.join(log_dir, epoch_dir_name,
                        f"{phase}-{epoch:0>3}-SimulatedImageMask_Prediction.png")
        )
        utils.draw_grid_map(
            images=[utils.segmap2rgbmap(item) 
                    for item in pred_masks
            ],             
            nrows=2,
            out_path=os.path.join(log_dir, epoch_dir_name,
                            f"{phase}-{epoch:0>3}-SimulatedMask_Prediction.png")
        )
        utils.draw_grid_map(
            images=real_images,
            nrows=2,
            out_path=os.path.join(log_dir, epoch_dir_name,
                                  f"{phase}-{epoch:0>3}-RealImages_Groundtruth.png")
        )
        utils.draw_grid_map(
            images=pred_images,
            nrows=2,
            out_path=os.path.join(log_dir, epoch_dir_name,
                                  f"{phase}-{epoch:0>3}-RealImages_Prediction.png")
        )
        utils.draw_grid_map(
            images=noisy_images,
            nrows=2,
            out_path=os.path.join(log_dir, epoch_dir_name,
                                  f"{phase}-{epoch:0>3}-NoisyRealImages.png")
        )