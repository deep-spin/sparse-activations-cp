import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import math

from transformers import AutoImageProcessor

from abc import ABC, abstractmethod

class Datasets(ABC):
    @abstractmethod
    def __init__(
            self,
            valid_ratio: float,
            batch_size: int,
            calibration_samples: int = 3000,
            ):
        pass
    
    @property
    def train(self):
        return self._train
    
    @property
    def dev(self):
        return self._dev
    
    @property
    def cal(self):
        return self._cal
    
    @property
    def test(self):
        return self._test

    def _train_splits(self, train_dataset,
                      calibration_samples, valid_ratio, batch_size):
        
        gen = torch.Generator()
        gen.manual_seed(0)
        
        train_dataset, cal_dataset = torch.utils.data.dataset.random_split(
            train_dataset, 
            [len(train_dataset)-calibration_samples, calibration_samples],
            generator=gen
        )

        nb_train = int(math.ceil((1.0 - valid_ratio) * len(train_dataset)))
        nb_valid = int(math.floor((valid_ratio * len(train_dataset))))
        train_dataset, dev_dataset = torch.utils.data.dataset.random_split(
            train_dataset, [nb_train, nb_valid], generator=gen
        )
        
        self._train = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self._dev = DataLoader(dev_dataset, batch_size=batch_size)
        self._cal = DataLoader(cal_dataset, batch_size=batch_size)