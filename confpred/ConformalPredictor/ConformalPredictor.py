import numpy as np
from abc import ABC, abstractmethod
from entmax.activations import entmax15, sparsemax
from entmax import entmax_bisect
import torch
from torch.nn.functional import softmax
from scipy.stats import rankdata

np.random.seed(0)
class ConformalPredictor():
    def __init__(self,score):
        self.score = score
    
    def calibrate(self, cal_true, cal_pred, alpha):
        n_cal = cal_true.shape[0]
        cal_scores = self.score.get_single_score(cal_true,cal_pred)
        self.cal_scores = cal_scores
        q_level = np.ceil((n_cal+1)*(1-alpha))/n_cal
        try:
            self.q_hat = np.quantile(cal_scores, q_level, method = 'higher')
        except TypeError:
            self.q_hat = np.quantile(cal_scores, q_level, interpolation = 'higher')
        self.p_values_cal = self.get_pvalue(cal_scores)
    def get_pvalue(self,preds):
        #return np.array([((self.cal_scores>= el).sum() + 1)/(len(self.cal_scores) + 1) for el in preds])
        sorted_cal_scores = np.sort(self.cal_scores)
        m = len(self.cal_scores)
        
        # Use searchsorted to find the number of elements greater than or equal to each element of preds
        greater_or_equal_count = m - np.searchsorted(sorted_cal_scores, preds, side='right')
        
        # Compute p-values
        p_values = (greater_or_equal_count + 1) / (m + 1)
        return p_values
    def predict(self, test_pred, disallow_empty = False,use_temperature = False):
        
        if use_temperature:
            if self.score.alpha == 1.5:
                qhat = self.q_hat
                pred_ent = entmax15((2/qhat)*torch.tensor(test_pred), dim=-1).numpy()
                test_match = pred_ent>0
            elif self.score.alpha == 2:
                qhat = self.q_hat
                pred_ent = sparsemax((1/qhat)*torch.tensor(test_pred), dim=-1).numpy()
                test_match = pred_ent>0
            elif self.score.alpha > 1:
                qhat = self.q_hat
                gamma = 1/(self.score.alpha-1)
                pred_ent = entmax_bisect((gamma/qhat)*torch.tensor(test_pred),
                                         alpha=self.score.alpha, dim=-1).numpy()
                test_match = pred_ent>0
            else:
                raise ValueError('Temperature only supported for alpha >1')
        else:
            test_scores = self.score.get_multiple_scores(test_pred)
            test_match = test_scores<= self.q_hat
            self.test_pvalues = np.apply_along_axis(self.get_pvalue,1,test_scores)
            self.test_scores = test_scores

        if disallow_empty:
            helper = np.zeros(test_pred[(test_match.sum(axis = 1)==0)].shape)
            helper[np.arange(helper.shape[0]),test_pred[(test_match.sum(axis = 1)==0)].argmax(axis = 1)]=1
            test_match[(test_match.sum(axis = 1)==0)] = helper
        return test_match
    
    def evaluate(self, test_true, test_pred, disallow_empty = False,use_temperature = False):
        n_test = test_pred.shape[0]
        test_match = self.predict(test_pred, disallow_empty,use_temperature)
        set_size = test_match.sum(axis = 1).mean()
        coverage = test_match[test_true.astype(bool)].sum()/n_test
        return set_size, coverage
    
class APSPredictor(ConformalPredictor):
    def predict(self, test_pred,disallow_empty = False,use_temperature = False):
        val_smx = self.score.get_multiple_scores(test_pred)
        val_pi = val_smx.argsort(1)[:, ::-1]
        val_srt = np.take_along_axis(val_smx, val_pi, axis=1).cumsum(axis=1)
        return np.take_along_axis(val_srt <= self.q_hat, val_pi.argsort(axis=1), axis=1)
    
class RAPSPredictor(ConformalPredictor):
    def predict(self, test_pred,disallow_empty = False,use_temperature = False):
        lam_reg = self.score.lam_reg
        k_reg = self.score.k_reg
        n_val = test_pred.shape[0]
        val_smx = softmax(torch.tensor(test_pred),dim=-1).numpy()
        val_pi = test_pred.argsort(1)[:,::-1]
        val_srt = np.take_along_axis(val_smx,val_pi,axis=1)
        reg_vec = np.array(k_reg*[0,] + (test_pred.shape[1]-k_reg)*[lam_reg,])[None,:]
        val_srt_reg = val_srt + reg_vec
        indicators = (val_srt_reg.cumsum(axis=1) - np.random.rand(n_val,1)*val_srt_reg) <= self.q_hat if self.score.rand else val_srt_reg.cumsum(axis=1) - val_srt_reg <= self.q_hat
        if disallow_empty: indicators[:,0] = True
        prediction_sets = np.take_along_axis(indicators,val_pi.argsort(axis=1),axis=1)
        return prediction_sets


class ConformalScore(ABC):
    
    @abstractmethod
    def get_single_score(self, cal_true, cal_pred) -> np.array:
        pass
    
    @abstractmethod
    def get_multiple_scores(self, test_pred) -> np.array:
        pass
    

class SparseScore(ConformalScore):
    def __init__(self, alpha):
        self.alpha = alpha
        
    def get_single_score(self, cal_true, cal_pred) -> np.array:
        # Rank data efficiently using scipy rankdata (assign ranks with 1 being the highest)
        ranks = rankdata(-cal_pred, axis=1, method='ordinal') - 1  # Reverse sort
        match = ranks[np.arange(ranks.shape[0]), cal_true.argmax(axis=1)]
        
        # Create a boolean mask where ranks are greater than the true class rank
        cond = ranks > match[:, None]
        
        # Get the predicted values for the true class (k_y)
        k_y = cal_pred[np.arange(cal_pred.shape[0]), cal_true.argmax(axis=1)]
        
        # Calculate the difference between predicted values and true class prediction
        output = cal_pred - k_y[:, None]
        
        # Zero out the values where the rank condition is met
        output[cond] = 0
        
        # Return the Lp-norm based on alpha
        return np.linalg.norm(output, axis=1, ord=1/(self.alpha - 1))
    def get_multiple_scores(self, test_pred) -> np.array:
        num_classes = test_pred.shape[1]
        batch_size = test_pred.shape[0]
        
        # Create a one-hot encoding for each possible class across the batch
        true_tests = np.eye(num_classes)[None, :, :]  # Shape: (1, num_classes, num_classes)
        
        # Compute scores for each class
        output = np.zeros((batch_size, num_classes))
        
        for i in range(num_classes):
            true_test = np.zeros_like(test_pred)
            true_test[:, i] = 1  # Set the i-th column as "true" class
            output[:, i] = self.get_single_score(true_test, test_pred)
            
        return output
""" class SparseScore(ConformalScore):
    def __init__(self, alpha):
        self.alpha = alpha 
        
     def get_single_score(self, cal_true, cal_pred) -> np.array:
        ranks = np.flip(cal_pred.argsort(axis = 1),axis = 1).argsort()
        match = np.select(cal_true.astype(bool).T,ranks.T)
        cond = ranks>np.expand_dims(match, axis=-1)
        k_y = np.select(cal_true.astype(bool).T,cal_pred.T)
        output = (cal_pred-np.expand_dims(k_y, axis=-1))
        output[cond] = 0
        return np.linalg.norm(output,axis = 1, ord = 1/(self.alpha-1)) 
    
     def get_multiple_scores(self, test_pred) -> np.array:
        output = []
        for i in range(test_pred.shape[1]):
            true_test = np.zeros(test_pred.shape)
            true_test[:,i] = 1
            output.append(self.get_single_score(true_test,test_pred)[None,:])
        return np.concatenate(output,axis=0).T """
    
class SoftmaxScore(ConformalScore):
    def get_single_score(self, cal_true, cal_pred) -> np.array:
        cal_sm = softmax(torch.tensor(cal_pred),dim=-1).numpy()
        true_mask = cal_true.astype(bool)
        cal_scores = 1 - cal_sm[true_mask]
        return cal_scores
    
    def get_multiple_scores(self, test_pred) -> np.array:
        return 1-softmax(torch.tensor(test_pred),dim=-1).numpy()
    
class APSScore(ConformalScore):
    def get_single_score(self, cal_true, cal_pred) -> np.array:
        n_cal = cal_true.shape[0]
        cal_sm = softmax(torch.tensor(cal_pred),dim=-1).numpy()
        cal_labels = cal_true.argmax(axis=1)
        cal_pi = cal_sm.argsort(1)[:,::-1]
        cal_srt = np.take_along_axis(cal_sm,cal_pi,axis=1).cumsum(axis=1)
        return np.take_along_axis(cal_srt,cal_pi.argsort(axis=1),axis=1)[range(n_cal),cal_labels]
    def get_multiple_scores(self, test_pred) -> np.array:
        return softmax(torch.tensor(test_pred),dim=-1).numpy()
    
class RAPSScore(ConformalScore):
    def __init__(self, lam_reg, k_reg,rand = True):
        self.lam_reg = lam_reg
        self.k_reg = k_reg
        self.rand = rand
    def get_single_score(self, cal_true, cal_pred) -> np.array:
        lam_reg = self.lam_reg
        k_reg = self.k_reg
        reg_vec = np.array(k_reg*[0,] + (cal_pred.shape[1]-k_reg)*[lam_reg,])[None,:]
        n = cal_true.shape[0]
        cal_labels = cal_true.argmax(axis=1)
        cal_smx = softmax(torch.tensor(cal_pred),dim=-1).numpy()
        cal_pi = cal_smx.argsort(1)[:,::-1]
        cal_srt = np.take_along_axis(cal_smx,cal_pi,axis=1)
        cal_srt_reg = cal_srt + reg_vec
        cal_L = np.where(cal_pi == cal_labels[:,None])[1]
        cal_scores = cal_srt_reg.cumsum(axis=1)[np.arange(n),cal_L] - np.random.rand(n)*cal_srt_reg[np.arange(n),cal_L]
        return cal_scores
        # Get the score quantile
    def get_multiple_scores(self, test_pred) -> np.array:
        return softmax(torch.tensor(test_pred),dim=-1).numpy()
    
import numpy as np
from scipy.stats import rankdata

class LimitScore(ConformalScore):
    def get_single_score(self, cal_true, cal_pred) -> np.array:
        # Rank predictions in descending order for each sample
        ranks = np.argsort(-cal_pred, axis=1)  # Get indices of descending order
        reverse_ranks = np.argsort(ranks, axis=1)  # Convert to ranks (0 = highest)
        
        # True class prediction scores
        k_y = cal_pred[np.arange(cal_pred.shape[0]), cal_true.argmax(axis=1)]
        
        # Get the largest score (highest ranked)
        largest = cal_pred[np.arange(cal_pred.shape[0]), ranks[:, 0]]
        
        return largest - k_y

    def get_multiple_scores(self, test_pred) -> np.array:
        num_classes = test_pred.shape[1]
        batch_size = test_pred.shape[0]
        
        # Initialize output matrix
        scores = np.zeros((batch_size, num_classes))
        
        for i in range(num_classes):
            # Assume class `i` is the true class
            true_test = np.zeros_like(test_pred)
            true_test[:, i] = 1  # Simulate the true class
            
            # Get scores for this assumed true class
            k_y = test_pred[np.arange(batch_size), i]  # Prediction for class i
            largest = np.max(np.where(np.arange(num_classes) != i, test_pred, -np.inf), axis=1)
            scores[:, i] = largest - k_y
        
        return scores
