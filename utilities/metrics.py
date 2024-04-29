import gc
import os
from collections import Counter
from enum import Enum
from typing import List, Dict
from typing import Union, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import torch
import torchmetrics
from monai.losses import DiceLoss, TverskyLoss, FocalLoss
from pytorch_msssim import MS_SSIM
from torch import nn
from torchvision import models

from .utils import AttrDict


class MeanMetric:
    """Calculate the mean of metrics.
    Args:
        scalar (bool): if True, calculate the mean of a single metric, else a dict of metrics.
    """
    def __init__(self, 
                 scalar: bool=False
    ) -> None:
        self.scalar = scalar
        self.reset()
        
    def reset(self
    ) -> None:
        if self.scalar is True:
            self.metric = 0
        else:
            self.metrics = {}
        self.count = 0

    def update(self, 
               metrics: Union[int, float, Dict]
    ) -> None:
        if self.scalar is True:
            self.metric += metrics
        else:
            self.metrics = dict(
                Counter(self.metrics) + Counter(metrics)
            )
        self.count += 1
        
    def compute(self
    ) -> AttrDict:
        if self.scalar is True:
            self.metrics = self.metric / self.count
        else:
            self.metrics = {
                k: v / self.count 
                for k, v in self.metrics.items()
            }
            self.metrics = AttrDict(self.metrics)
        return self.metrics


class ClassificationLoss(torch.nn.Module):
    """Calculate loss by combining different losses for a batch of data.
    Args:
        task (str): the task of the classification mode.
            `binary` or `multi` mode is used to preprocess the model output for
            caluclating the loss.
    Return:
        loss (Tensor): the torch.Tensor calculated loss.
    """

    def __init__(self,
                 task: str = 'multiclass'
    ) -> None:
        super(ClassificationLoss, self).__init__()

        self.task = task
        if self.task == 'multiclass':
            self.ce = torch.nn.CrossEntropyLoss()
        elif self.task == 'binary':
            self.ce = torch.nn.BCEWithLogitsLoss()
        else:
            raise ValueError('Only `binary` or `multiclass` classification task modes are supported.')

    def forward(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        preds = preds.squeeze()
        if self.task == 'binary':
            labels = labels.float()
        loss = self.ce(preds, labels)
        return loss


class ContrastiveLoss(torch.nn.Module):
    def __init__(self,
                 tau: float = 0.1
    ) -> None:
        super(ContrastiveLoss, self).__init__()
        self.tau = tau

    def forward(self,
                pos_features: torch.Tensor,
                neg_features: torch.Tensor
    ) -> torch.Tensor:
        pos_features = torch.div(pos_features, torch.norm(pos_features, dim=1, keepdim=True))
        neg_features = torch.div(neg_features, torch.norm(neg_features, dim=1, keepdim=True))
        cat12_feat = torch.cat([pos_features, neg_features], dim=0)
        cat21_feat = torch.cat([neg_features, pos_features], dim=0)
        similarities = torch.exp(torch.mm(cat12_feat, cat12_feat.t()) / self.tau)
        rows_sum = torch.sum(similarities, dim=1)
        diag = torch.diag(similarities)
        numerators = torch.exp(
                torch.div(
                    torch.nn.CosineSimilarity()(cat12_feat, cat21_feat),
                    self.tau
                )
        )
        denominators = rows_sum - diag
        contrastive_loss = torch.mean(
                    -1 * torch.log(
                            torch.div(numerators, denominators)
                    )
        )
        return contrastive_loss


class BinaryMetrics(Enum):
    ACCURACY = torchmetrics.classification.BinaryAccuracy
    PRECISION = torchmetrics.classification.BinaryPrecision
    RECALL = torchmetrics.classification.BinaryRecall
    F1SCORE = torchmetrics.classification.BinaryF1Score


class MulticlassMetrics(Enum):
    ACCURACY = torchmetrics.classification.MulticlassAccuracy
    PRECISION = torchmetrics.classification.MulticlassPrecision
    RECALL = torchmetrics.classification.MulticlassRecall
    F1SCORE = torchmetrics.classification.MulticlassF1Score


class ClassificationMetrics:
    """Calculate the classification scores such as `accuracy`, `precision`,
        `recall`, and `f1-score`.
    Args:
        task (str): The binary or non-binary classification task.
            `binary` or `multiclass` mode is used to preprocess the model output for
            caluclating the loss.
        num_classes (int): The number of classes in the classification task.
        classes (sequence): The name of the classess to be used in visualizing the
            confusion matrix.
        average (str): This parameter is required for multiclass/multilabel tasks.
            The options are ‘micro’, ‘macro’, ‘weighted’, ‘none’.
            More information about each option in `TorchMetrics`.
        sigmoid (bool): set it to `True` for binary classifiers.
        save_confmatrix (bool): set it to `True` if you want to calculate
            and save confusion matrix.
        conf_out_dir (str): The directory address for saving the confusion matrices.
        confmat_normalize (str): The normalization method for the confusion matrix. Options are 'true', 'none'.
    Returns:
        scores (Dict): a dictionary contains the scores. Each key is the score name and its value
            would the flaoting point score value.
    """

    def __init__(self,
                 task: str = 'multiclass',
                 num_classes: int = 3,
                 classes: Union[Tuple[str], List[str]] = ('class0', 'class1', 'class2'),
                 average: str = 'micro',
                 sigmoid: bool = False,
                 binary_threshold: float = 0.5,
                 save_confmatrix: bool = False,
                 conf_out_dir: str = 'confusion_matrix.png',
                 confmat_normalize: str = 'true'
    ) -> None:
        self.task = task
        self.num_classes = num_classes
        self.classes = classes
        self.average = average
        self.sigmoid = sigmoid
        self.binary_threshold = binary_threshold
        self.save_confmatrix = save_confmatrix
        self.conf_out_dir = conf_out_dir
        self.confmat_normalize = confmat_normalize

        if self.task == 'binary':
            self.metric_functions = {
                met.name: met.value(threshold=self.binary_threshold)
                for met in BinaryMetrics
            }
            self.confmat = torchmetrics.classification.BinaryConfusionMatrix(
                threshold=self.binary_threshold,
                normalize=self.confmat_normalize
            )
        elif self.task == 'multiclass':
            self.metric_functions = {
                met.name: met.value(num_classes=self.num_classes, average=self.average)
                for met in MulticlassMetrics
            }
            self.confmat = torchmetrics.classification.MulticlassConfusionMatrix(
                num_classes=self.num_classes,
                normalize=self.confmat_normalize
            )
        else:
            raise ValueError('Only `binary` or `multiclass` classification task modes are supported.')

        # Define the collection to save the input data.
        self.reset()

    def update(self,
               preds: torch.Tensor,
               targets: torch.Tensor
    ) -> Dict:
        """Update the prediction and targest collection by the new batch data."""
        preds, targets = self.format_labels(preds, targets)
        self.preds_collection = torch.hstack([self.preds_collection, preds])
        self.targets_collection = torch.hstack([self.targets_collection, targets])

    def reset(self
              ) -> None:
        """Reset the collections for the new collection process of the new epoch."""
        self.preds_collection = torch.tensor([], dtype=torch.short)
        self.targets_collection = torch.tensor([], dtype=torch.short)

    def close(self
              ) -> None:
        """Remove the collections and free the memory."""
        del self.preds_collection
        del self.targets_collection
        gc.collect()

    def compute_scores(self
                       ) -> Dict:
        """Compute the defined scores for the collected predictisons/targets."""
        assert len(self.preds_collection) > 0, 'The prediction collection is empty.'
        assert len(self.targets_collection) > 0, 'The target collection is empty.'
        scores = {}
        for key, func in self.metric_functions.items():
            scores[key] = func(self.preds_collection,
                               self.targets_collection
                               ).item()
        return scores

    def compute_confmatrix(self,
                           conf_file_name: str
                           ) -> None:
        """Build and save the confusion matrix as an image in the provided output path."""
        assert len(self.preds_collection) > 0, 'The prediction collection is empty.'
        assert len(self.targets_collection) > 0, 'The target collection is empty.'
        if self.save_confmatrix is True:
            cf_matrix = self.confmat(
                self.preds_collection,
                self.targets_collection
            ).cpu().numpy()
            cf_df = pd.DataFrame(
                cf_matrix,
                index=[i for i in self.classes],
                columns=[i for i in self.classes]
            )
            plt.figure(figsize=(10, 10))
            sn.heatmap(cf_df, annot=True, annot_kws={"size": 24})
            plt.xlabel('Predicted', fontsize=24, color='tab:red')
            plt.ylabel('Actual', fontsize=24, color='tab:cyan')
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.savefig(os.path.join(self.conf_out_dir, conf_file_name), dpi=1200)
        else:
            print('Warning: Confusion matrix calculator has not been activated.')

    def format_labels(self,
                      preds: torch.Tensor,
                      targets: torch.Tensor
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert the predicted probablities to labels and correct the prediction and targets
            formats based on the `binary` or `multiclass` tasks.
        """
        preds = preds.clone().cpu()
        targets = targets.clone().cpu()
        if self.sigmoid is True and self.task == 'binary':
            preds = torch.ge(torch.sigmoid(preds), self.binary_threshold)
        elif self.sigmoid is False and self.task == 'binary':
            preds = torch.ge(preds, self.binary_threshold)
        else:
            _, preds = torch.max(preds, dim=1)
        preds = preds.flatten().type(torch.short)
        targets = targets.flatten().type(torch.short)
        return preds, targets

    def format_metrics(self,
                       scores: Dict
                       ) -> str:
        """Format the calucalted metrics into a printable string message."""
        formatted = []
        for key, val in scores.items():
            formatted.append(f'{key}: {val: 0.4f}')
        return ', '.join(formatted)


class SegmentationLoss(torch.nn.Module):
    def __init__(self,
                 loss_names: List = ['bce']
    ) -> None:
        super(SegmentationLoss, self).__init__()
        self.loss_names = loss_names
        self.loss_functions = {
            'bce': nn.BCEWithLogitsLoss(),
            'dice': DiceLoss(sigmoid=True),
            'tversky': TverskyLoss(sigmoid=True),
            'focal': FocalLoss(gamma=2.0)
        }

    def forward(self, 
                output: torch.Tensor, 
                target: torch.Tensor
    )-> torch.Tensor:
        target = target.float()
        loss = self.loss_functions[self.loss_names[0]](output, target)
        for i in range(1, len(self.loss_names)):
            loss += self.loss_functions[self.loss_names[i]](output, target)
        return loss


class SegmentationMetrics(object):
    def __init__(self, apply_sigmoid=True, threshold=0.5):
        self.apply_sigmoid = apply_sigmoid
        self.threshold = threshold

    def __call__(self, inputs, targets, smooth=1.0):
        with torch.no_grad():
            if inputs.shape != targets.shape:
                raise ValueError("inputs and targets should have same shapes.")
            if len(inputs.shape) != 4:
                raise ValueError("targets and inputs shape should be of length 4.")
            if self.apply_sigmoid is True:
                y_pred = torch.ge(torch.sigmoid(inputs), self.threshold).float()
            else:
                y_pred = inputs.float()
            y_true = targets.float()

            # reducing only spatial dimensions (not batch nor channels)
            n_len = len(y_pred.shape)
            reduce_axis = list(range(2, n_len))
            intersection = torch.sum(
                torch.logical_and(y_true, y_pred), dim=reduce_axis
            )
            union = torch.sum(
                torch.logical_or(y_true, y_pred), dim=reduce_axis
            )
            true_segment = torch.sum(y_true, dim=reduce_axis)
            pred_segment = torch.sum(y_pred, dim=reduce_axis)
            dice = (2 * intersection + smooth) / (true_segment + pred_segment + smooth)
            iou = (intersection + smooth) / (union + smooth)
            channel_dice = torch.mean(dice, dim=0)
            channel_iou = torch.mean(iou, dim=0)
            scores = {
                'iou': torch.mean(channel_iou).item(),
                'dice': torch.mean(channel_dice).item(),
                # 'channel_iou': channel_iou.cpu().numpy(),
                # 'channel_dice': channel_dice.cpu().numpy()
            }
        return scores
    

class PerceptualLoss(torch.nn.Module):
    def __init__(self, 
                 device: str = 'cpu'
    ) -> None: 
        super().__init__()
        # self.per_net = models.vgg16(weights="DEFAULT").features
        self.per_net = nn.Sequential(
            *list(
                models.resnet18(weights="DEFAULT").children()
            )[:-2]
        )
        self.per_net = self.per_net.to(device)
        # self.per_net.requires_grad = False
        self.per_net.eval()
        self.perceptual_loss_func = torch.nn.L1Loss(reduction="mean")

    def forward(self, 
                prediction: torch.Tensor, 
                groundtruth: torch.Tensor
    ) -> torch.Tensor:
        prediction_features = self.per_net(prediction)
        with torch.no_grad():
            groundtruth_features = self.per_net(groundtruth)
        perceptual_loss = self.perceptual_loss_func(
            prediction_features, 
            groundtruth_features
        )
        return perceptual_loss
    

class ReconstructionLoss(torch.nn.Module):
    def __init__(self, 
                 loss_names: List = ['mse', 'mae', 'ssim', 'perceptual'],
                 device: str = 'cpu',
    ) -> None:
        super().__init__()
        self.loss_names = loss_names  
        self.device = device      
        self.loss_functions = {
            'mse': torchmetrics.regression.MeanSquaredError().to(self.device), 
            'mae': torchmetrics.regression.MeanAbsoluteError().to(self.device), 
            'ssim': MS_SSIM(size_average=True, channel=3).to(self.device), 
            'perceptual': PerceptualLoss(device=self.device).to(self.device)
        }
        
    def forward(self,
                prediction: torch.Tensor,
                groundtruth: torch.Tensor
    ) -> torch.Tensor:
        loss_value = self.loss_functions[self.loss_names[0]](
            prediction, 
            groundtruth
        )
        for loss_func in self.loss_names[1:]:
            loss_value += self.loss_functions[loss_func](
                prediction, 
                groundtruth
            )
        return loss_value
    
    def __repr__(self):
        return f"{self.__class__.__name__}(loss_names={self.loss_names}, device={self.device})"
