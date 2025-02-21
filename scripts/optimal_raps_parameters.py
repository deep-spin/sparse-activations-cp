import pickle
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from confpred import RAPSPredictor, RAPSScore
from pathlib import Path

ROOT_DIR = Path('../')
# print the full path name
print(ROOT_DIR.resolve())



model_loss = 'softmax'
transformation = 'logits'
dataset_list = ['NewsGroups','CIFAR100','ImageNet','CIFAR10']
seed = '23'
random_state_list = [1,12,123,1234,12345]
k_list = [1,5,10,50]
lam_reg_list = [0.001,0.01,0.1,1]
alpha_list = np.round(np.linspace(0.01,0.1,10),4)
summary_results = pd.DataFrame({'dataset':dataset_list})\
    .merge(pd.DataFrame({'random_state':random_state_list}), how = 'cross')\
    .merge(pd.DataFrame({'lam_reg':lam_reg_list}), how = 'cross')\
        .merge(pd.DataFrame({'alpha':alpha_list}), how = 'cross')\
            .merge(pd.DataFrame({'k_reg':k_list}), how = 'cross')
summary_results['avg_size'] = np.nan
summary_results['coverage'] = np.nan
summary_results.set_index(['dataset','random_state','lam_reg','alpha','k_reg'],inplace=True)
summary_results.sort_index(inplace=True)

for dataset in dataset_list:
    if dataset=='NewsGroups':
        model_type = 'bert'
    else:
        model_type = 'vit'
    path = f'{ROOT_DIR}/data/predictions/{model_type}_{dataset}_test_{model_loss}_{transformation}_{seed}_proba.pickle'
    with open(path, 'rb') as f:
        test_preds_og = pickle.load(f)
    path = f'{ROOT_DIR}/data/predictions/{dataset}_{seed}_test_true.pickle'
    with open(path, 'rb') as f:
        test_true_enc_og = pickle.load(f)
    for random_state in random_state_list:
        cal_size = np.ceil(test_true_enc_og.shape[0]*0.4).astype(int)
        print(dataset+'_'+str(random_state))
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

        #print(dataset+'_'+str(param)+'_'+str(seed)+'_'+str(alpha)+'_'+str(random_state))
        for k in k_list:
            if k<test_true_enc.shape[1]:
                for lam_reg in lam_reg_list:
                    for alpha in alpha_list:
                        print(dataset+'_'+str(lam_reg)+'_'+str(k)+'_'+str(seed)+'_'+str(alpha)+'_'+str(random_state))
                        cp = RAPSPredictor(RAPSScore(lam_reg,k))
                        cp.calibrate(cal_true_enc_1, cal_proba_1, alpha)
                        avg_size, coverage = cp.evaluate(cal_true_enc_2, cal_proba_2)
                        summary_results.loc[(dataset,random_state,lam_reg,alpha,k),'avg_size'] = avg_size
                        summary_results.loc[(dataset,random_state,lam_reg,alpha,k),'coverage'] = coverage

summary_results.reset_index(inplace=True)
valid_results = summary_results#[summary_results['coverage']>=1-summary_results['alpha']]
valid_results.sort_values(by='avg_size',ascending=True,inplace=True)
final_params = valid_results.groupby(['dataset','alpha','random_state']).head(1)
final_params.sort_values(by='dataset',inplace=True)
final_params['param'] = 'RAPS_optimal'

final_params.to_csv('../data/results_analysis/raps_optimal_parameters.csv',index=False)