"""Joao Calem - FineTuneViT.py"""

from transformers import AutoModelForImageClassification
import torch.nn as nn

from .BaseClassifier import BaseClassifier

class FineTuneViT(BaseClassifier):
    def __init__(self,
            n_classes: int,
            transformation: str = 'softmax',
            ):
        """
        Constructor for CNN model
        """
        
        super().__init__(transformation) 

        self.vit = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.classifier = nn.Linear(self.vit.classifier.in_features, n_classes)
            
    def __get_logits__(self, x):
        """
        Forward pass for specified transformation function on intitialisation.
        """
        
        return self.vit(x)[0]