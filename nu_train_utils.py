# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 19:11:14 2019 @author: berdakh.abibullaev
"""
import numpy as np
import time
import copy
import torch
from torch.utils.data import TensorDataset, DataLoader
import pdb 

################
dev = torch.device("cpu")

if torch.cuda.is_available():
    dev = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
################
def best_epoch_labels(train_labels, best_epoch):        
    for jj in range(len(train_labels[best_epoch]['ypred'])-1):    
        if jj == 0: 
          ypred = train_labels[best_epoch]['ypred'][jj]              
          ytrue = train_labels[best_epoch]['ytrue'][jj]                  
        ypred = np.concatenate([ypred, train_labels[best_epoch]['ypred'][jj+1]])
        ytrue = np.concatenate([ytrue, train_labels[best_epoch]['ytrue'][jj+1]])            
    return ypred, ytrue

###############
def train_model(model, dset_loaders, dset_sizes, criterion, optimizer, dev,
                lr_scheduler=None, num_epochs=50, verbose=2, LSTM = False):    
    
      start_time = time.time()        
      best_model, best_acc = model, 0.0 
      train_losses,val_losses,train_accs,val_accs,train_labels,val_labels=[],[],[],[],[],[]
      
      for epoch in range(num_epochs):     
          if verbose > 1: print('Epoch {}/{}'.format(epoch+1, num_epochs))          
          ypred_labels, ytrue_labels = [], [] 

          # [Train or Validation phase]           
          for phase in ['train', 'val']:
              if phase == 'train':
                  if lr_scheduler: optimizer = lr_scheduler(optimizer, epoch)
                  model.train(True)   
              else:
                  model.train(False)   
                    
              # Iterate over mini-batches       
              batch, running_loss, running_corrects = 0.0, 0.0, 0.0            
            
              for data in dset_loaders[phase]:
                  inputs, labels = data                  
                  optimizer.zero_grad()        
                    
                  if LSTM:                                      
                      hidden = model.init_hidden(len(inputs.to(dev)))                        
                      preds = model(inputs.to(dev), hidden)                      
                  else:  
                      preds = model(inputs.to(dev)) 
                      
                  labels = labels.type(torch.LongTensor)    
                  loss   = criterion(preds, labels.to(dev))  
                  
                  # Backpropagate & weight update 
                  if phase == 'train':
                      loss.backward()
                      optimizer.step()                                               
                  # store batch performance 
                  with torch.no_grad():                      
                      running_loss     += float(loss.item())                  
                      running_corrects += torch.sum(preds.data.max(1)[1] ==
                                                    labels.to(dev).data)                     
                      ytrue_labels.append(labels.data.cpu().detach().numpy())
                      ypred_labels.append(preds.data.max(1)[1].cpu().detach().numpy())                    
                      batch += 1
                      del loss
                      
              with torch.no_grad():                      
                  epoch_loss = running_loss / dset_sizes[phase]
                  epoch_acc  = running_corrects.cpu().numpy()/dset_sizes[phase]         
    
                  if phase == 'train':
                      train_losses.append(epoch_loss)
                      train_accs.append(epoch_acc)                   
                      train_labels.append(dict(ypred = ypred_labels, ytrue = ytrue_labels))                     
                  else: 
                      val_losses.append(epoch_loss)
                      val_accs.append(epoch_acc)                
                      val_labels.append(dict(ypred = ypred_labels, ytrue = ytrue_labels))
    
                  if verbose > 1: print('{} loss: {:.4f}, acc: {:.4f}'.format(phase, 
                                                                              epoch_loss,
                                                                              epoch_acc))
            
                  # Deep copy the best model using early stopping
                  if phase == 'val' and epoch_acc > best_acc:
                      best_acc = epoch_acc
                      best_epoch = epoch 
                      best_model = copy.deepcopy(model)           
             
      time_elapsed = time.time() - start_time  
      
      # ytrue and ypred from the best model during the training    
      ytrain_best = best_epoch_labels(train_labels, best_epoch)
      yval_best   = best_epoch_labels(val_labels,   best_epoch)      
      
      info = dict(ytrain = ytrain_best, yval = yval_best, 
                  best_epoch = best_epoch, best_acc = best_acc)
      
      if verbose > 0:
          print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, 
                                                              time_elapsed % 60))
          print('Best val Acc: {:4f}'.format(best_acc)) 
          print('Best Epoch :', best_epoch+1) 
            
      return best_model, train_losses, val_losses, train_accs, val_accs, info
#################