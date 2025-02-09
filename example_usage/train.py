import argparse

from confpred.classifier import load_model, train, evaluate
from confpred.datasets import load_dataset
from entmax.losses import SparsemaxLoss, Entmax15Loss
import json
import torch
from torch import nn
from sklearn.metrics import f1_score
import numpy as np
import random 
import os

def run_train(model_type, dataset, loss, save_name, seed, epochs, patience):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}", flush=True)
    
    device = 'cuda:1' if torch.cuda.is_available() and torch.cuda.device_count()>1 else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    if loss == 'sparsemax':
        criterion = SparsemaxLoss()
    elif loss == 'softmax':
        criterion = torch.nn.NLLLoss()
    elif loss == 'entmax':
        criterion = Entmax15Loss()

    data = load_dataset(dataset, model_type)
    
    model = load_model(dataset, model_type, loss, device)
    
    model, train_history, val_history, f1_history = train(model,
                                                data.train,
                                                data.dev,
                                                criterion,
                                                epochs=epochs,
                                                patience=patience,
                                                device=device)

    _, predicted_labels, true_labels, test_loss = evaluate(
                                                        model,
                                                        data.test,
                                                        criterion,
                                                        device=device)

    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    print(f'Test loss: {test_loss:.3f}')
    print(f'Test f1: {f1:.3f}')

    results = {
        'train_history':train_history,
        'val_history':val_history,
        'f1_history':f1_history,
    }

    with open(f'{save_name}_config.json', 'w') as f:
        json.dump(results, f)
        
    torch.save(model.state_dict(), f'{save_name}_model.pth')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train model.")

    parser.add_argument('model', type=str,
                        help='The name of the classification  model')

    parser.add_argument('dataset', type=str,
                        help='The name of the dataset')

    parser.add_argument('loss', type=str,
                        help='The loss funtion to use for training')
    
    parser.add_argument('save', type=str,
                        help='The file name to save the model')
    
    parser.add_argument('--seed', type=int, help='The seed to use for training', 
                                default=123)
    
    parser.add_argument('--epochs', type=int, help='The maximum number of epochs', 
                                default=300)
    
    parser.add_argument('--patience', type=int, help='The training patience', 
                                default=3)

    args = parser.parse_args()
    
    run_train(args.model, args.dataset, args.loss, args.save, args.seed,
              args.epochs, args.patience)