"""Joao Calem - FineTuneBertForSequenceClassification.py"""

from transformers import BertForSequenceClassification, AdamW, BertConfig
import torch.nn as nn

from .BaseClassifier import BaseClassifier

class FineTuneBertForSequenceClassification(BaseClassifier):
    def __init__(self,
            n_classes: int,
            transformation: str = 'softmax',
            ):
        """
        Constructor for CNN model
        """
        
        super().__init__(transformation) 

        self.bert = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=n_classes,  # Number of labels (20) corresponds to the 20 newsgroups dataset
                output_attentions=False,  # Do not output attention weights
                output_hidden_states=False,  # Do not output hidden states
        )
            
    def __get_logits__(self, input_ids, attention_mask):
        """
        Forward pass for specified transformation function on intitialisation.
        """
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).logits