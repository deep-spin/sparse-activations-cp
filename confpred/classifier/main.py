from confpred.utils import EarlyStopper

from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score

device = 'cuda:1' if torch.cuda.is_available() and torch.cuda.device_count()>1 else 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate(model, dataloader, criterion, device=device):
    
    model.eval()
    
    pred_proba=[]
    pred_labels=[]
    true_labels = []
    losses = []
    
    to_numpy = lambda x: x.numpy() if "cpu" in device else x.detach().cpu().numpy()
    with torch.no_grad():
        for _, data in tqdm(enumerate(dataloader, 0),total=len(dataloader)):
            data = [d.to(device) for d in data]
            inputs = data[:-1]
            labels = data[-1]
            with torch.no_grad():
                outputs = model(*inputs)
            pred_proba.append(to_numpy(outputs))
            pred_labels.append(to_numpy(outputs.argmax(dim=-1)))
            true_labels.append(to_numpy(labels))
            
            losses.append(criterion(outputs, labels))
                
    pred_proba = np.concatenate(pred_proba)
    pred_labels = np.concatenate(pred_labels)
    true_labels = np.concatenate(true_labels)
    
    loss = torch.tensor(losses).mean().item()
    
    model.train()

    return pred_proba, pred_labels, true_labels, loss

def train(model,
        train_dataloader,
        dev_dataloader,
        criterion,
        epochs=15,
        patience=3,
        device=device):

    early_stopper = EarlyStopper(patience=patience)
    total_steps = len(train_dataloader) * epochs

    # Create the optimizer and scheduler for fine-tuning the model
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    train_history = []
    val_history = []
    f1_history = []
    
    for epoch in range(epochs):  # loop over the dataset multiple times

        print(f'-- Epoch {epoch + 1} --')
              
        train_losses = []
            # zero the parameter gradients
            
        progress_bar = tqdm(train_dataloader, desc="Training", position=0, leave=True)
        for data in progress_bar:
            # get the inputs; data is a list of [inputs, labels]
            data = [d.to(device) for d in data]
            inputs = data[:-1]
            labels = data[-1]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(*inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()

            # print statistics
            train_losses.append(loss)
            
            progress_bar.set_description(f"Training - Loss: {loss.item():.4f}")
        
            
            
        train_loss = torch.tensor(train_losses).mean().item()
        print(f'train_loss: {train_loss:.3f}')
            
        _, predicted_labels, true_labels, val_loss = evaluate(model,
                                                        dev_dataloader,
                                                        criterion,
                                                        device=device)
        print(f'val_loss: {val_loss:.3f}')
        
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        print(f'val_f1: {f1:.3f}')
        
        train_history.append(train_loss)
        val_history.append(val_loss)
        f1_history.append(f1)
        
        final_model = early_stopper.early_stop(val_loss, model)
        if final_model:             
            break
    if not final_model:
        final_model = model
        
    print('-- Finished Training --')
    return final_model, train_history, val_history, f1_history
