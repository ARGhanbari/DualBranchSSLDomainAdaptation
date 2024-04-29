import torch 
from torch import nn 

@torch.no_grad()
def init_weights(
    method: str = "kaiming_normal",
    mean: float = 0.0,
    std: float = 0.5,
    low: float = 0.0,
    high: float = 1.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    gain: float = 1.0
):
    """Initialize the network's weights based on the provided method

    Args:
        m: (nn.Module) module itself
        method: (str) how to initialize the weights
        mean: (float) mean of normal distribution
        std: (float) standard deviation for normal distribution
        low: (float) minimum threshold for uniform distribution
        high: (float) maximum threshold for uniform distribution
        mode: (str) either 'fan_in' (default) or 'fan_out'. Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes in the backwards pass.
        nonlinearity: (str) the non-linear function (nn.functional name), recommended to use only with 'relu' or 'leaky_relu' (default).
        gain: (float) an optional scaling factor for xavier initialization
    """
    if method == "kaiming_normal":
        def init(m):
            if any(
                [
                    isinstance(m, nn.Conv1d),
                    isinstance(m, nn.Conv2d),
                    isinstance(m, nn.Conv3d),
                    isinstance(m, nn.Linear),
                ]
            ):
                nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity=nonlinearity)
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / (fan_out ** 0.5)
                    nn.init.normal_(m.bias, -bound, bound)
            # Kaiming_init does not work with Batchnorm
            elif any(
                [
                    isinstance(m, nn.BatchNorm1d),
                    isinstance(m, nn.BatchNorm2d),
                    isinstance(m, nn.BatchNorm3d),
                ]
            ):
                nn.init.normal_(m.weight, mean=mean, std=std)
        return init
    
    # kaiming_uniform_
    elif method == "kaiming_uniform_":
        def init(m):
            if any(
                [
                    isinstance(m, nn.Conv1d),
                    isinstance(m, nn.Conv2d),
                    isinstance(m, nn.Conv3d),
                    isinstance(m, nn.Linear),
                ]
            ):
                nn.init.kaiming_uniform_(m.weight, mode=mode, nonlinearity=nonlinearity)
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / (fan_out ** 0.5)
                    nn.init.uniform_(m.bias, -bound, bound)
            # Kaiming_init does not work with Batchnorm
            elif any(
                [
                    isinstance(m, nn.BatchNorm1d),
                    isinstance(m, nn.BatchNorm2d),
                    isinstance(m, nn.BatchNorm3d),
                ]
            ):
                nn.init.uniform_(m.weight, mean=mean, std=std)
        return init
    
    # normal
    elif method == "normal":
        def init(m):
            if any(
                [
                    isinstance(m, nn.Conv1d),
                    isinstance(m, nn.Conv2d),
                    isinstance(m, nn.Conv3d),
                    isinstance(m, nn.BatchNorm1d),
                    isinstance(m, nn.BatchNorm2d),
                    isinstance(m, nn.BatchNorm3d),
                    isinstance(m, nn.Linear),
                ]
            ):

                nn.init.normal_(m.weight, mean=mean, std=std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        return init
    
    # uniform
    elif method == "uniform":
        def init(m):
            if any(
                [
                    isinstance(m, nn.Conv1d),
                    isinstance(m, nn.Conv2d),
                    isinstance(m, nn.Conv3d),
                    isinstance(m, nn.BatchNorm1d),
                    isinstance(m, nn.BatchNorm2d),
                    isinstance(m, nn.BatchNorm3d),
                    isinstance(m, nn.Linear),
                ]
            ):
                nn.init.uniform_(m.weight, a=low, b=high)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        return init
    
    # xavier_normal
    elif method == "xavier_normal":
        def init(m):
            if any(
                [
                    isinstance(m, nn.Conv1d),
                    isinstance(m, nn.Conv2d),
                    isinstance(m, nn.Conv3d),
                    isinstance(m, nn.BatchNorm1d),
                    isinstance(m, nn.BatchNorm2d),
                    isinstance(m, nn.BatchNorm3d),
                    isinstance(m, nn.Linear),
                ]
            ):
                nn.init.xavier_normal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        return init
    
    # xavier_uniform
    elif method == "xavier_uniform":
        def init(m):
            if any(
                [
                    isinstance(m, nn.Conv1d),
                    isinstance(m, nn.Conv2d),
                    isinstance(m, nn.Conv3d),
                    isinstance(m, nn.BatchNorm1d),
                    isinstance(m, nn.BatchNorm2d),
                    isinstance(m, nn.BatchNorm3d),
                    isinstance(m, nn.Linear),
                ]
            ):
                nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        return init