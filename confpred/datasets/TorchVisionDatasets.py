from .Datasets import Datasets
from confpred.utils import ROOT_DIR

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

from transformers import AutoImageProcessor

from abc import ABC, abstractmethod

class TorchVisionDatasets(Datasets, ABC):
    def __init__(
            self,
            valid_ratio: float,
            batch_size: int,
            calibration_samples: int = 3000,
            transform: str = 'norm'
            ):

        train_dataset = self._get_dataset(transform, train=True)
        self._train_splits(train_dataset,
                           calibration_samples,valid_ratio, batch_size)

        test_dataset = self._get_dataset(transform, train=False)
        self._test = DataLoader(test_dataset, batch_size=batch_size)
        
        if transform=='vit':
            self.vit_processor = AutoImageProcessor.from_pretrained(
                            "google/vit-base-patch16-224",use_fast=True)
    
    @abstractmethod
    def _dataset_class(self):
        pass

    def _dataset(self, transform):
        data_class, normalize = self._dataset_class()
        
        if transform == 'norm':
            transform = transforms.Compose(
                [transforms.ToTensor(),
                normalize])
        elif transform == 'vit':
            transform = lambda x: self.vit_processor(x.convert('RGB'))['pixel_values'][0]
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        
        return data_class, transform
    
    def _get_dataset(self, norm, train=True):
        data_class, transform = self._dataset(norm)
        return data_class(
            root=os.path.join(ROOT_DIR,'data'),
            train=train,
            download=True,
            transform=transform
        )