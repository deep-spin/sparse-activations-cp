from .Datasets import Datasets

import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from tqdm import tqdm
import math

class NewsGroups(Datasets):
    def __init__(
            self,
            valid_ratio: float,
            batch_size: int,
            calibration_samples: int = 3000,
            test_ratio: float = 0.2
            ):
        
        newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
        data = pd.DataFrame({'text_data': newsgroups.data, 'label': newsgroups.target})

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        max_seq_len = 128
        
        input_ids, attention_masks, labels = self._tokenize_data(data, tokenizer, max_seq_len)

        dataset = TensorDataset(input_ids, attention_masks, labels)

        gen = torch.Generator()
        gen.manual_seed(0)
        
        nb_train = int(math.ceil((1.0 - test_ratio) * len(dataset)))
        nb_test = int(math.floor((test_ratio * len(dataset))))
        train_dataset, test_dataset = torch.utils.data.dataset.random_split(
            dataset, [nb_train, nb_test], generator=gen
        )
        
        self._train_splits(train_dataset,
                           calibration_samples,valid_ratio, batch_size)
        
        self._test = DataLoader(test_dataset, batch_size=batch_size)
        
        
    def _tokenize_data(self, data, tokenizer, max_seq_len):
        input_ids, attention_masks, labels = [], [], []

        # Iterate through each row in the dataset
        for index, row in tqdm(data.iterrows(), total=len(data)):
            # Tokenize the text using BERT's tokenizer with additional parameters
            encoded = tokenizer.encode_plus(
                row["text_data"],
                add_special_tokens=True,  # Add [CLS] and [SEP] tokens
                max_length=max_seq_len,  # Set max sequence length to 128
                padding="max_length",  # Pad shorter sequences to max_seq_len
                truncation=True,  # Truncate longer sequences to max_seq_len
                return_attention_mask=True,  # Return attention masks
            )

            # Append tokenized data to respective lists
            input_ids.append(encoded["input_ids"])
            attention_masks.append(encoded["attention_mask"])
            labels.append(row["label"])

        # Convert lists to tensors
        return torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(labels)