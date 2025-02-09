from .TorchVisionDatasets import TorchVisionDatasets
from confpred.utils import ROOT_DIR

import torchvision
import torchvision.transforms as transforms
import os

class ImageNet(TorchVisionDatasets):
    def __init__(
            self,
            valid_ratio: float,
            batch_size: int,
            calibration_samples: int = 3000,
            norm: bool = True
            ):
        super().__init__(valid_ratio, batch_size, calibration_samples, norm)
    
    def _dataset_class(self):
        data_class = torchvision.datasets.ImageNet
        normalize = transforms.Normalize(0.5, 0.5, 0.5)
        return data_class, normalize
    
    def _get_dataset(self, norm, train=True):
        split = 'val'
        if train:
            split='train'
        data_class, transform = self._dataset(norm)
        return data_class(
            root=os.path.join(ROOT_DIR,'data','imagenet'),
            split=split,
            transform=transform
        )