import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from confpred import ConformalPredictor,SparseScore
import pickle
from pathlib import Path

ROOT_DIR = Path('../')
# print the full path name
print(ROOT_DIR.resolve())

dataset_list = ['CIFAR10','CIFAR100','NewsGroups','ImageNet']
seed = '23'
model_loss = 'softmax'
alpha_list = np.round(np.linspace(0.01,0.1,10),4)
random_states = [1,12,123,1234,12345]
ROOT_DIR = '..'
score_list = ['sparsemax','softmax','entmax']
transformation='logits'
lambda_list = [1.1,1.3,1.5,1.7,1.9]
summary_results = pd.DataFrame({'dataset':dataset_list})\
        .merge(pd.DataFrame({'alpha':alpha_list}), how = 'cross')\
            .merge(pd.DataFrame({'random_state':random_states}), how = 'cross')
summary_results['avg_size'] = np.nan
summary_results['coverage'] = np.nan
summary_results['best_lambda'] = np.nan
summary_results.set_index(['dataset','random_state','alpha'],inplace=True)
summary_results.sort_index(inplace=True)
for dataset in dataset_list:
    if dataset == 'NewsGroups':
        model_type = 'bert'
    elif dataset == 'CIFAR10':
        model_type = 'cnn'
    else:
        model_type = 'vit'
    path = f'{ROOT_DIR}/data/predictions/{model_type}_{dataset}_test_{model_loss}_{transformation}_{seed}_proba.pickle'
    with open(path, 'rb') as f:
        test_preds_og = pickle.load(f)
    path = f'{ROOT_DIR}/data/predictions/{dataset}_{seed}_test_true.pickle'
    with open(path, 'rb') as f:
        test_true_enc_og = pickle.load(f)
    for alpha in alpha_list:
        print(f'Running {dataset} with alpha = {alpha}')
        for random_state in random_states:
            cal_sizes = np.zeros(len(lambda_list))
            
            cal_size = np.ceil(test_true_enc_og.shape[0]*0.4).astype(int)
            test_preds,test_true_enc = shuffle(test_preds_og,test_true_enc_og,random_state = random_state)
            cal_proba = test_preds[0:cal_size]
            test_proba = test_preds[cal_size:]
            cal_true_enc = test_true_enc[0:cal_size]
            test_true_enc = test_true_enc[cal_size:]
            
            # Split for hyperparameter change
            cal_size_2 = np.ceil(cal_size*0.6).astype(int)
            cal_proba_1 = cal_proba[0:cal_size_2]
            cal_true_enc_1 = cal_true_enc[0:cal_size_2]
            cal_proba_2 = cal_proba[cal_size_2:]
            cal_true_enc_2 = cal_true_enc[cal_size_2:]
            for i, l in enumerate(lambda_list):
                cp = ConformalPredictor(SparseScore(alpha=l))
                cp.calibrate(cal_true_enc_1,cal_proba_1,alpha)
                size, _ = cp.evaluate(
                                        cal_true_enc_2,cal_proba_2,
                                        disallow_empty=False, 
                                        use_temperature=True)
                cal_sizes[lambda_list.index(l)] = size
            best_l = lambda_list[cal_sizes.argmin()]
            cp = ConformalPredictor(SparseScore(alpha=best_l))
            cp.calibrate(cal_true_enc_1,cal_proba_1,alpha)
            avg_size, coverage = cp.evaluate(
                                             test_true_enc,test_proba,
                                             disallow_empty=False, 
                                             use_temperature=True)
            summary_results.loc[(dataset,random_state,alpha),'avg_size'] = avg_size
            summary_results.loc[(dataset,random_state,alpha),'coverage'] = coverage
            summary_results.loc[(dataset,random_state,alpha),'best_lambda'] = best_l
            
summary_results.reset_index().to_csv(f'{ROOT_DIR}/data/results_analysis/optimal_entmax_parameters.csv',index=False)