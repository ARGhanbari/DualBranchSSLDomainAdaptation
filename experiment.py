import argparse
import os
import shutil

import pandas as pd
import torch

import development
import dual_unet
import utilities as utils
from data import get_dataloader
from diffusion import SimpleDiffusion


def main() -> None:
    # Input arguments.
    parser = argparse.ArgumentParser(description='Initializers parameters for running the experiments.')
    parser.add_argument('-c', '--config', dest='config_path', type=str, default='configs/ldm.yaml',
                    help='The string path of the config file.')
    args = parser.parse_args()
    
    configs = utils.ConfigLoader()(config_path=args.config_path)
    
    torch.cuda.set_device(0)
    configs.BasicConfig.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Sut up output dirs.
    if configs.BasicConfig.DEVELOPMENT_PHASE != "PRED": 
        log_dir, checkpoint_dir = utils.setup_log_directory(config=configs.BasicConfig)
    if configs.BasicConfig.DEVELOPMENT_PHASE == 'TEST':
        log_dir = os.path.join(log_dir, configs.BasicConfig.TESTEXPERIMENTDIRNAME)
        checkpoint_dir = os.path.join(checkpoint_dir, configs.BasicConfig.TESTEXPERIMENTDIRNAME)
        os.makedirs(name=log_dir, exist_ok=True)
        os.makedirs(name=checkpoint_dir, exist_ok=True)
    if configs.BasicConfig.DEVELOPMENT_PHASE != "PRED":     
        shutil.copy(f"{args.config_path}", log_dir)
        shutil.copy('autoencoder.py', log_dir)
        shutil.copy('utilities/transformations.py', log_dir)

    # Set the seeds.
    utils.set_seeds(
        configs.BasicConfig.SEED,
        configs.BasicConfig.DETERMINISTIC_CUDANN,
        configs.BasicConfig.BENCHMARK_CUDANN, 
        configs.BasicConfig.DETERMINISTIC_ALGORITHM
    )

    # Set the logger.
    wandb_logger = None
    if configs.BasicConfig.LOGGER == 'wandb' and configs.BasicConfig.DEVELOPMENT_PHASE == 'TRAIN':
        wandb_logger = utils.WeightAndBiases(
            project_name=configs.BasicConfig.PROJECT_NAME,
            experiment_name=configs.BasicConfig.EXPERIMENT_NAME,
            entity=configs.BasicConfig.ENTITY,
            configs=configs
        )
    # Load DataSet and DataLoader
    transformations = utils.get_transformations(configs.TrainConfig)
    if configs.BasicConfig.DEVELOPMENT_PHASE == 'TRAIN':
        real_train_loader = get_dataloader(
            configs.TrainConfig.REAL_TRAIN_METADATA_PATH,
            configs.TrainConfig.REAL_TRAIN_SUBSET_SIZE,
            transformations['REAL_TRAIN'],
            configs.TrainConfig, 
            phase='TRAIN', 
            seed=configs.BasicConfig.SEED
        )
        simu_train_loader = get_dataloader(
            configs.TrainConfig.SIMU_TRAIN_METADATA_PATH,
            configs.TrainConfig.SIMU_TRAIN_SUBSET_SIZE,
            transformations['SIMU_TRAIN'],
            configs.TrainConfig,
            phase='TRAIN', 
            seed=configs.BasicConfig.SEED
        )
        real_valid_loader = get_dataloader(
            configs.TrainConfig.REAL_VALID_METADATA_PATH,
            -1, # subset_size
            transformations['REAL_VALID'],
            configs.TrainConfig,
            phase='VALID', 
            seed=configs.BasicConfig.SEED
        )
        simu_valid_loader = get_dataloader(
            configs.TrainConfig.SIMU_VALID_METADATA_PATH,
            -1, # subset_size
            transformations['SIMU_VALID'],
            configs.TrainConfig,
            phase='VALID', 
            seed=configs.BasicConfig.SEED
        )
    elif configs.BasicConfig.DEVELOPMENT_PHASE == 'TEST':
        test_loader = get_dataloader(
            configs.TrainConfig.TEST_METADATA_PATH,
            -1, # subset_size
            transformations['TEST'],
            configs.TrainConfig,
            phase='TEST', 
            seed=configs.BasicConfig.SEED
        )
    elif configs.BasicConfig.DEVELOPMENT_PHASE == 'PRED':
        pred_loader = get_dataloader(
            configs.TrainConfig.PRED_METADATA_PATH,
            -1, # subset_size
            transformations['PRED'],
            configs.TrainConfig,
            phase='PRED', 
            seed=configs.BasicConfig.SEED
        )
            
    # Define the diffusion sampler.
    diffusion_sampler = SimpleDiffusion(
        num_diffusion_timesteps=configs.TrainConfig.TIMESTEPS,
        img_shape=configs.TrainConfig.IMG_SHAPE,
        device=configs.BasicConfig.DEVICE
    )
    # Models definition.
    model = dual_unet.get_unet(
        configs,
        wandb_logger
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=configs.TrainConfig.LR,
        weight_decay=configs.TrainConfig.WD
    )
    # Criteria and Metrics.
    img_loss_fn = utils.ReconstructionLoss( 
        loss_names=configs.TrainConfig.IMAGE_LOSSES, 
        device=configs.BasicConfig.DEVICE
    )
    print(
        'Image Loss: ', 
        img_loss_fn
    )
    msk_loss_fn = utils.SegmentationLoss(
        loss_names=configs.TrainConfig.MASK_LOSSES
    )
    msk_score_fn = utils.SegmentationMetrics()
    if configs.BasicConfig.DEVELOPMENT_PHASE == 'TRAIN':
        loaders_dict = {
            'TRAIN': {'REAL': real_train_loader,
                      'SIMU': simu_train_loader
            },
            'VALID': {'REAL': real_valid_loader,
                      'SIMU': simu_valid_loader
            }
        }
        trainer = development.Development(
            model=model,
            loaders=loaders_dict,
            diffusion_sampler=diffusion_sampler,
            diffusion_patch_size=configs.TrainConfig.DIFFUSION_PATCH_SIZE,
            time_steps=configs.TrainConfig.ADDEDNOISETIMESTEPS,
            optimizer=optimizer,
            img_loss_fn=img_loss_fn,
            msk_loss_fn=msk_loss_fn,
            msk_score_fn=msk_score_fn,
            epochs=configs.TrainConfig.NUM_EPOCHS,
            device=configs.BasicConfig.DEVICE,
            log_dir=log_dir,
            logger=wandb_logger,
            inverse_mean=configs.TrainConfig.TRANSFORM_MEAN,
            inverse_std=configs.TrainConfig.TRANSFORM_STD
        )
    
        best_dice = 0.0
        best_rec_loss = float('inf')
        scores_df = None
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(1, configs.TrainConfig.NUM_EPOCHS + 1):
            print()
            # torch.cuda.empty_cache()
            # gc.collect()
            # ======== TRAIN ========== #
            model.train()
            epoch_train_loss, rec_train_loss, seg_train_loss, epoch_train_metrics = trainer(
                phase='TRAIN',                
                epoch=epoch
            )
            # ======== VALID ========== #
            model.eval()
            epoch_valid_loss, rec_valid_loss, seg_valid_loss, epoch_valid_metrics = trainer(
                phase='VALID',                
                epoch=epoch
            )
            # ======== LOGGING ========== #
            epoch_score_dict = {
                f"Epoch": epoch,
                f"Train Loss": epoch_train_loss,
                f"Train RecLoss": rec_train_loss,
                f"Train SegLoss": seg_train_loss,
                f"Train Dice": epoch_train_metrics['dice'],
                f"Train IoU": epoch_train_metrics['iou'],
                f"Valid Loss": epoch_valid_loss,
                f"Valid RecLoss": rec_valid_loss,
                f"Valid SegLoss": seg_valid_loss,
                f"Valid Dice": epoch_valid_metrics['dice'],
                f"Valid IoU": epoch_valid_metrics['iou']
            }
            if wandb_logger is not None:
                wandb_logger.log(epoch_score_dict)
            if scores_df is None:
                scores_df = pd.DataFrame(epoch_score_dict, index=[0])
            else:
                scores_df = pd.concat([scores_df, pd.DataFrame(epoch_score_dict, index=[0])], 
                                    ignore_index=True, sort=False, axis=0
                )
            # ======== MODEL MONITORING ========== #
            # Save each model.
            checkpoint_dict = {
                'epoch': epoch,
                "opt": optimizer.state_dict(),
                "model": model.state_dict()
            }
            if rec_valid_loss < best_rec_loss:
                best_rec_loss = rec_valid_loss
                torch.save(checkpoint_dict,
                        os.path.join(checkpoint_dir, f'Rec_{epoch:0>4}.pt')
                )
            # Save the best model. 
            if epoch_valid_metrics['dice'] > best_dice:
                best_dice = epoch_valid_metrics['dice']
                torch.save(checkpoint_dict,
                    os.path.join(checkpoint_dir, f'Seg_{epoch:0>4}.pt')
                )
                print(f'Best Model Saved in Epoch {epoch:0>3}.')
                
            torch.save(checkpoint_dict,
                os.path.join(checkpoint_dir, f'Epoch_{epoch:0>4}.pt')
            )
            del checkpoint_dict

            # Load another subset of data after each epoch.  
            loaders_dict["TRAIN"]["REAL"].dataset.subset_sampler()
            loaders_dict["TRAIN"]["SIMU"].dataset.subset_sampler()

        scores_df.to_csv(os.path.join(log_dir, 'organized_scores.csv'), index=False)
    elif configs.BasicConfig.DEVELOPMENT_PHASE == 'TEST': # (Only for autoencoders evaluation)
        tester = development.Development(
            model=model,
            loaders={'TEST': {'REAL': test_loader,
                              'SIMU': test_loader
                              }
            },
            diffusion_sampler=diffusion_sampler,
            diffusion_patch_size=configs.TrainConfig.DIFFUSION_PATCH_SIZE,
            time_steps=configs.TrainConfig.ADDEDNOISETIMESTEPS,
            optimizer=None,
            img_loss_fn=img_loss_fn,
            msk_loss_fn=msk_loss_fn,
            msk_score_fn=msk_score_fn,
            epochs=0,
            device=configs.BasicConfig.DEVICE,
            log_dir=log_dir,
            logger=wandb_logger,
            inverse_mean=configs.TrainConfig.TRANSFORM_MEAN,
            inverse_std=configs.TrainConfig.TRANSFORM_STD
        )
        # ======== TEST ========== #
        model.eval()
        _, _, _, _ = tester(phase='TEST', epoch=0)
    else: # Phase == PRED
        print('prediction ... '.upper())
        print('-' * 50)
        os.makedirs(
            configs.BasicConfig.PREDICTION_DIR, 
            exist_ok=True
        )
        model.eval()
        development.Development.predict(
            model=model, 
            loaders={
                'REAL': pred_loader,
                'SIMU': pred_loader
            }, 
            device=configs.BasicConfig.DEVICE, 
            inverse_mean=configs.TrainConfig.TRANSFORM_MEAN,
            inverse_std=configs.TrainConfig.TRANSFORM_STD,
            prediction_dir=configs.BasicConfig.PREDICTION_DIR, 
        )

if __name__ == '__main__':
    main()