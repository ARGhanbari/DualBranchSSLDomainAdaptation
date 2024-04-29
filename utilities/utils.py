import json
import logging
import os
import random
import sys
from typing import Union, Optional, Dict

import numpy as np
import torch
import wandb


class AttrDict(Dict):
    """AttrDict class to handle both index and attribute access to a dictionary.
    Args:
        records: dictionary of records.
    Returns:
        None
    """
    def __init__(self, 
                 records: Dict
    ) -> None:
        super().__init__()
        self.records = records
        for key, value in self.records.items():
            if isinstance(value, dict):
                setattr(self, key, AttrDict(value))
            else:
                setattr(self, key, value)
                            
    def __getitem__(self, name):
        return self.records[name]
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
        self.records[key] = value
    
    def keys(self):
        return self.records.keys()
    
    def values(self):
        return self.records.values()
    
    def items(self):
        return self.records.items()
    
    def update(self, other):
        self.records.update(other)
        for key, value in other.items():
            setattr(self, key, value)
    
    def __iter__(self):
        return iter(self.records)
    
    def __len__(self):
        return len(self.records)
    
    def __getattr__(self, name):
        if name in self.records:
            return self.records[name]
        raise AttributeError(f"Attribute {name} not found")
    
    def __repr__(self) -> str:
        return f"AttrDict({self.records})"


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


class Logger(object):
    def __init__(self,
                 log_dir: str = 'out/',
                 log_file_name: str = 'experiment.log',
                 log_mode: str = 'a', 
                 add_stream_handler: bool = True,
                 stream_terminator: str = '\n'
    ) -> None:
        self.log_dir = log_dir
        self.log_file_name = log_file_name
        self.log_mode = log_mode
        self.add_stream_handler = add_stream_handler
        self.stream_terminator = stream_terminator
        
        handlers = [
            logging.FileHandler(os.path.join(self.log_dir, self.log_file_name), 
                                self.log_mode)
        ]
        if self.add_stream_handler:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.terminator = self.stream_terminator
            handlers.append(stream_handler)
        
        logging.basicConfig(
            format='',
            level=logging.INFO,
            handlers=handlers
        )
        self.logger = logging.getLogger('')

    def log(self,
            message: str
    ) -> None:
        self.logger.info(message)

    def log_dict(self,
                 message: Dict
    ) -> None:
        self.logger.info(
            json.dumps(message, indent=4, sort_keys=True)
        )

    def log_newline(self
    ) -> None:
        self.logger.info('\n')

    def log_separator(self
    ) -> None:
        self.logger.info('-' * 50)


class WeightAndBiases(object):
    """Weight and biases logger.
    Args:
        project_name: name of the project
        experiment_name: name of the experiment
        api_key: API key for weight and biases
    Returns:
        None
    """
    def __init__(self,
                 project_name: str,
                 experiment_name: str,
                 entity: str = 'default',
                 configs: Union[AttrDict, Dict] = None
    ) -> None:
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.entity = entity
        self.configs = configs
        
        # Initialize wandb.
        wandb.init(
            project=self.project_name,
            name=self.experiment_name,
            entity=self.entity,
            config=configs
        )
        self.logger = wandb

    def watch(self,
              model
    ) -> None:
        self.logger.watch(model)

    def log(self,
            message: Union[str, Dict]
    ) -> None:
        if isinstance(message, str):
            message = {'message': message}
        self.logger.log(message)

    def log_image(self,
                  images: Optional[Union[torch.Tensor, np.ndarray]],
                  predictions: Optional[Union[torch.Tensor, np.ndarray]]=None,
                  ground_truth: Optional[Union[torch.Tensor, np.ndarray]]=None,
                  class_labels: Optional[Dict]=None
    ) -> None:
        """Log images with or without ground truth to weight and biases."""
        if class_labels is None:
            class_labels = {0: 'Background', 1: 'Foreground'}
        
        if isinstance(images, torch.Tensor):
            images = images.permute(0, 2, 3, 1).numpy()
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.permute(0, 2, 3, 1).numpy()
        if isinstance(ground_truth, torch.Tensor):
            ground_truth = ground_truth.permute(0, 2, 3, 1).numpy()
        
        predictions = [
            self.logger.Image(
                images[i],
                masks={
                    'predictions': {
                        'mask_data': np.zeros(images[i].shape[0:-1], dtype=np.uint8) if predictions is None else predictions[i].squeeze(),
                        'class_labels': {0: 'Background'} if predictions is None else class_labels
                    },
                    'ground_truth': {
                        'mask_data': np.zeros(images[i].shape[0:-1], dtype=np.uint8) if ground_truth is None else ground_truth[i].squeeze(),
                        'class_labels': {0: 'Background'} if ground_truth is None else class_labels
                    }
                }
            )
            for i in range(images.shape[0])
        ]
        self.logger.log({
            'predictions': predictions
        })


def make_message(metrics: Dict,
                 sep=', '
) -> str:
    """Make a message from the metrics dictionary.
    Args: 
        metrics (Dict): dictionary of metrics. 
        sep (str): separator for the message.
    Returns:
        message (str): string message.
    """
    elements = reversed(sorted(metrics.keys()))
    s = []
    for k in elements:
        value = metrics[k]
        if np.isscalar(value):
            s.append('{}: {:0.4f}'.format(k, value))
        else:
            s.append(
                '{}: [{}]'.format(k, sep.join([f'{x:0.4f}' for x in value]))
            )
    return sep.join(s)

@torch.no_grad()
def check_grad_norm(
        model: torch.nn.Module
) -> float:
    """Compute and return the grad norm of all parameters of the network.
    To see gradients flowing in the network or not.
    Args: 
        model (torch.nn.Module): model to check the grad norm.
    """
    total_norm = 0
    for p in list(filter(lambda p: p.grad is not None, model.parameters())):
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm

def get_default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_tensor_element(element: torch.Tensor, 
        t: torch.Tensor
) -> torch.Tensor:
    """
    Get value at index position "t" in "element" and
        reshape it to have the same dimension as a batch of images.
    """
    ele = element.gather(-1, t)
    # (B, C, H, W)
    return ele.reshape(-1, 1, 1, 1)

def setup_log_directory(config: Union[Dict, AttrDict]
) -> None:
    '''Log and Model checkpoint directory Setup'''
    last_version_number = config.EXPERIMENT_NAME.split('_')[-1]
    if (config.DEVELOPMENT_PHASE == 'TRAIN' and 
        os.path.isdir(config.ROOT_LOG_DIR)
    ):
        folder_numbers = [
            int(folder.split("_")[-1])
            for folder in os.listdir(config.ROOT_LOG_DIR)
        ]
        last_version_number = max(folder_numbers) + 1
    # Define experiment dir. 
    version_name = f"{''.join(config.EXPERIMENT_NAME.split('_')[:-1])}_{last_version_number:0>3}" 

    # Update the training config default directory
    log_dir        = os.path.join(config.ROOT_LOG_DIR,        version_name)
    checkpoint_dir = os.path.join(config.ROOT_CHECKPOINT_DIR, version_name)

    # Create new directory for saving new experiment version
    os.makedirs(log_dir,        exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Logging at: {log_dir}")
    print(f"Model Checkpoint at: {checkpoint_dir}")

    return log_dir, checkpoint_dir

def to_device(data,
              device: torch.device
) -> torch.Tensor:
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def set_seeds(seed: int,
              deter_cnn: bool = True,
              benchm_cnn: bool = False, 
              deter_algorithm: bool = True
) -> None:
    # Set random seeds.
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deter_cnn
    torch.backends.cudnn.benchmark = benchm_cnn
    torch.use_deterministic_algorithms(deter_algorithm)      