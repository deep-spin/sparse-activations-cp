from confpred import ConformalPredictor,SparseScore,SoftmaxScore
from confpred.classifier import load_model, evaluate
from confpred.datasets import load_dataset
from confpred.utils import ROOT_DIR

from entmax.losses import SparsemaxLoss, Entmax15Loss
import torch
import numpy as np
import pandas as pd 
import os.path
import pickle

def run_cp(dataset, loss, alpha, seed, model_type='cnn', epochs=20, disallow_empty = False, use_temperature = False, model_loss = None):
    if model_loss is None:
        model_loss = loss
    
    #loss = 'softmax' #sparsemax, softmax or entmax15
    transformation = 'logits'
    #dataset='CIFAR100' #CIFAR100 or MNIST

    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    
    fname = f'{ROOT_DIR}/data/predictions/{model_type}_{dataset}_test_{model_loss}_{transformation}_{seed}_proba.pickle'
    if os.path.isfile(fname):
        print('Loading predictions.')
        path = f'{ROOT_DIR}/data/predictions/{model_type}_{dataset}_test_{model_loss}_{transformation}_{seed}_proba.pickle'
        with open(path, 'rb') as f:
            test_proba = pickle.load(f)
        path = f'{ROOT_DIR}/data/predictions/{dataset}_{seed}_test_true.pickle'
        with open(path, 'rb') as f:
            test_true_enc = pickle.load(f)
        path = f'{ROOT_DIR}/data/predictions/{model_type}_{dataset}_cal_{model_loss}_{transformation}_{seed}_proba.pickle'
        with open(path, 'rb') as f:
            cal_proba = pickle.load(f)
        path = f'{ROOT_DIR}/data/predictions/{dataset}_{seed}_cal_true.pickle'
        with open(path, 'rb') as f:
            cal_true_enc = pickle.load(f)
    else:
        data = load_dataset(dataset, model_type)
    
        model = load_model(dataset, model_type, transformation, device)
        model.load_state_dict(torch.load(f'{ROOT_DIR}/models/training/{model_type}_{dataset}_{loss}_{seed}_{epochs}_model_model.pth', map_location=torch.device(device)))
        print('Running predictions.')
        if loss == 'sparsemax':
            criterion = SparsemaxLoss()
        elif loss == 'softmax':
            criterion = torch.nn.NLLLoss()
        elif loss== 'entmax':
            criterion = Entmax15Loss()
            
        test_proba, _, test_true, _ = evaluate(
                                        model,
                                        data.test,
                                        criterion,
                                        device=device)

        cal_proba, _, cal_true, _ = evaluate(
                                        model,
                                        data.cal,
                                        criterion,
                                        device=device)

    #One Hot Encoding
        test_true_enc = np.zeros((test_true.size, test_true.max()+1), dtype=int)
        test_true_enc[np.arange(test_true.size),test_true] = 1

        cal_true_enc = np.zeros((cal_true.size, cal_true.max()+1), dtype=int)
        cal_true_enc[np.arange(cal_true.size),cal_true] = 1
        
        predictions = {'test':{'proba':test_proba,'true':test_true_enc},
                       'cal':{'proba':cal_proba,'true':cal_true_enc}}
        for dataset_type in ['cal','test']:
            with open(f'{ROOT_DIR}/data/predictions/{dataset}_{seed}_{dataset_type}_true.pickle', 'wb') as f:
                pickle.dump(predictions[dataset_type]['true'], f)
            with open(
                f'{ROOT_DIR}/data/predictions/{model_type}_{dataset}_{dataset_type}_{loss}' +
                    f'_{transformation}_{seed}_{"proba"}.pickle'
                , 'wb'
            ) as f:
                pickle.dump(predictions[dataset_type]["proba"], f)
    #Conformal Prediction
    if loss == 'sparsemax':
        cp = ConformalPredictor(SparseScore(2))
    elif loss == 'softmax':
        cp = ConformalPredictor(SoftmaxScore())
    elif loss== 'entmax':
        cp = ConformalPredictor(SparseScore(1.5))
    
    cp.calibrate(cal_true_enc, cal_proba, alpha)
    avg_set_size, coverage = cp.evaluate(test_true_enc, test_proba, disallow_empty, use_temperature)
    
    return avg_set_size, coverage

