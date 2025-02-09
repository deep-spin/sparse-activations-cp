"""Joao Calem - CNN.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

from .BaseClassifier import BaseClassifier

class CNN(BaseClassifier):
    """
    CNN Model for softmax or sparsemax transformations.

    Parameters
    ----------
    n_classes: int
        Number of target classes.
    input_size: int
        Input width of image.
    channels: int
        Number of channels in input image.
    transformation: str = 'softmax'
        Transformation applied to end of model forward pass.
    conv_channels: List[int] = [8,16]
        List of variable size with number of channels at each convulation layer.
    ffn_hidden_size: int = 32
        Size of hidden layer in feed forward network (FFN).
    kernel: int = 3
        Kernel size of convolutional layers.
    padding: int = 1
        Padding applied to convolutional layers.
    convs_per_pool: int = 1
        Number of convolutional layers done for every max pooling layer.
    batch_norm: bool = False
        Whether batch normalisation is applied after convolutions and in FFN
        
    Methods
    -------
    forward:
        Forward pass for specified transformation function on intitialisation.
    train:
        Set model to training mode.
    eval:
        Set model to evaluation mode.

    Returns
    -------
    None
    """
    def __init__(self,
            n_classes: int,
            input_size: int,
            channels: int,
            transformation: str = 'softmax',
            conv_channels: List[int] = [8,16],
            ffn_hidden_size: int = 32,
            kernel: int = 3,
            padding: int = 1,
            convs_per_pool: int = 1,
            batch_norm: bool = False):
        """
        Constructor for CNN model
        """
        
        super().__init__(transformation) 
        
        self._convs_per_pool = convs_per_pool
        self._convs = nn.ModuleList([])
        self._batch_norms=nn.ModuleList([])
        self._pool = nn.MaxPool2d(2, 2)
        self._dropout = nn.Dropout(0.2)
        self._b1d = None
        
        self._setup_convolutions(input_size, padding, kernel, channels, 
            conv_channels, convs_per_pool, batch_norm)
         
        self._fc1 = nn.Linear((self._shape**2)*conv_channels[-1], ffn_hidden_size)
        if batch_norm:
            self._b1d = nn.BatchNorm1d(ffn_hidden_size,0.005,0.95)
        self._fc2 = nn.Linear(ffn_hidden_size, n_classes)
    
    def _setup_convolutions(self, input_size, padding, kernel,
            channels, conv_channels, convs_per_pool, batch_norm):
        """
        Sets up structure of convolution and pooling layers of CNN.
        Saves output shape of these layers in self._shape.
        """
        
        channel_previous = channels
        self._shape = input_size
        size_adjust = 2*padding-kernel+1
        
        for channels in conv_channels:
            for i in range(convs_per_pool):
                self._convs.append(nn.Conv2d(channel_previous,
                                            channels,
                                            kernel,
                                            padding=padding
                                            ))
                if batch_norm:
                    self._batch_norms.append(nn.BatchNorm2d(channels))
                channel_previous=channels
                self._shape += size_adjust
            self._shape = self._shape//2
    
    def __get_logits__(self, x):
        """
        Forward pass for specified transformation function on intitialisation.
        """
        
        for i in range(0,len(self._convs),self._convs_per_pool):
            for j in range(self._convs_per_pool):
                if self._batch_norms:
                    x = F.relu(self._batch_norms[i+j](self._convs[i+j](x)))
                else:
                    x = F.relu(self._convs[i+j](x))
            x = self._dropout(self._pool(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        
        x = self._dropout(F.relu(self._fc1(x)))
        if self._b1d:
            x = self._b1d(x)
            
        return self._fc2(x)
        
