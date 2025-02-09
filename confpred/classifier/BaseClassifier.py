"""Joao Calem - BaseClassifier.py"""

from abc import ABC, abstractmethod
import torch.nn as nn
from entmax import sparsemax, entmax15
from typing import final



class BaseClassifier(ABC, nn.Module):
    def __init__(
            self,
            transformation: str = 'softmax',
            ):
        
        nn.Module.__init__(self)
        
        self._transformation = transformation
        self.train()
    
    @abstractmethod
    def __get_logits__(self,x):
        pass
       
    @final    
    def forward(self, *inputs):
        x = self.__get_logits__(*inputs)
        return self._final(x)
        
        
    def eval(self):
        """
        Set model to evaluation mode.
        """
        
        super().eval()
        if self._transformation=='softmax':
            self._final = lambda x: nn.Softmax(-1)(x)
        elif self._transformation=='sparsemax':
            self._final = lambda x: sparsemax(x,-1)
        elif self._transformation=='entmax':
            self._final = lambda x: entmax15(x,-1)
        elif self._transformation=='logits':
            self._final = lambda x: x
            
    def train(self, mode=True):
        """
        Set model to training mode.
        """
        
        super().train(mode)
        if self._transformation=='softmax':
            self._final = lambda x: nn.LogSoftmax(-1)(x)
        else:
            self._final = lambda x: x