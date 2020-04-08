#%%
### import all of the required modules
import itertools
import numpy as np
import time, copy, pdb
import pandas as pd 
import mne
import os 

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.init import constant_, xavier_normal_
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.utils import shuffle 

from train_utils_looV2 import (NDStandardScaler, train_model_without_val, subject_specific, leave1out, CNN1D)

dev = torch.device("cpu")

if torch.cuda.is_available():
    dev = torch.device("cuda")
    print(torch.cuda.get_device_name())
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
torch.backends.cudnn.benchmark = True    

import pickle

import pandas
from train_utils_bci import subject_specific, pad_with_zeros, pad_with_zeros_below, concatenate_array, concatenate_dicts

from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# roc curve and auc score
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


class CNN2D(torch.nn.Module):  
    """ Flexible 2D CNN 
    Example Usage:
        from nu_models import CNN_2DMod
        model = CNN_2DMod(kernel_size = [3, 3, 3, 3] , conv_channels = [1, 8, 16, 32])    
    """
    def __init__(self, input_size, 
                 kernel_size, 
                 conv_channels,
                 dense_size,
                 dropout  ):    
        
        super(CNN2D, self).__init__()          
        self.cconv   = []  
        self.MaxPool = nn.MaxPool2d((1, 2), (1, 2))  
#        self.MaxPool = nn.MaxPool2d(2, 2, padding = 1)
        self.ReLU    = nn.ReLU()
        self.Dropout = nn.Dropout(dropout)        
        self.batchnorm = []        
        
        for jj in conv_channels:
            self.batchnorm.append(nn.BatchNorm2d(jj, eps=0.001, momentum=0.01,
                                                 affine=True, track_running_stats=True).cuda())               
        ii = 0        
        # define CONV layer architecture:    
        for in_channels, out_channels in zip(conv_channels, conv_channels[1:]):                           
            conv_i = torch.nn.Conv2d(in_channels  = in_channels, 
                                     out_channels = out_channels,
                                     kernel_size  = kernel_size[ii],
#                                     stride       = (1, 2),
                                     padding      = (kernel_size[ii][0]//2, 
                                                     kernel_size[ii][1]//2)
                                     )
            
            self.cconv.append(conv_i)                
            self.add_module('CNN_K{}_O{}'.format(kernel_size[ii], out_channels), conv_i)
            ii += 1            
            
        self.flat_fts = self.get_output_dim(input_size, self.cconv)    
        self.fc1 = torch.nn.Linear(self.flat_fts, dense_size)
        self.fc2 = torch.nn.Linear(dense_size, 2)        
        
    def get_output_dim(self, input_size, cconv):        
        with torch.no_grad():
            input = torch.ones(1,*input_size)              
            for conv_i in cconv:                
                input = conv_i(input)
                input = self.MaxPool(input)        
                flatout = int(np.prod(input.size()[1:]))
                print(input.shape)
                print("Flattened output ::", flatout)                
        return flatout 

    def forward(self, input):        
        for jj, conv_i in enumerate(self.cconv):
            conv_i.cuda()
            
            input = conv_i(input)
            input = self.batchnorm[jj+1](input)
            input = self.ReLU(input)        
            input = self.MaxPool(input)    
               
        # flatten the CNN output     
        out = input.view(-1, self.flat_fts) 
        out = self.fc1(out)                       
        out = self.Dropout(out)        
        out = self.fc2(out)      
        return out        


def train_model(model, dset_loaders, dset_sizes, criterion, optimizer, dev,
                lr_scheduler=None, num_epochs=50, verbose=2, LSTM = False):   
      patience=30
    
    
      early_stopping = EarlyStopping(patience=patience, verbose=True)
      """
      Method to train a PyTorch neural network with the given parameters for a
      certain number of epochs. Keeps track of the model yielding the best validation
      accuracy during training and returns that model before potential overfitting
      starts happening. Records and returns training and validation losses and accuracies over all
      epochs.

      Args:
          model (torch.nn.Module): The neural network model that should be trained.

          dset_loaders (dict[string, DataLoader]): Dictionary containing the training
              loader and test loader: {'train': trainloader, 'val': testloader}
          dset_sizes (dict[string, int]): Dictionary containing the size of the training
              and testing sets. {'train': train_set_size, 'val': test_set_size}

          criterion (PyTorch criterion): PyTorch criterion (e.g. CrossEntropyLoss)
          optimizer (PyTorch optimizer): PyTorch optimizer (e.g. Adam)

          lr_scheduler (PyTorch learning rate scheduler, optional): PyTorch learning rate scheduler
          num_epochs (int): Number of epochs to train for
          verbose (int): Verbosity level. 0 for none, 1 for small and 2 for heavy printouts
      """    

      start_time = time.time()
      best_model, best_acc = model, 0.0
      val_corrects = np.zeros([num_epochs,])

      train_losses, val_losses, train_accs, val_accs  = [], [], [], []
      train_labels, val_labels = [], []

      for epoch in range(num_epochs):    
          if verbose > 1: print('Epoch {}/{}'.format(epoch+1, num_epochs))          
          ypred_labels, ytrue_labels = [], []          
          # train phase
          phase = 'train'
          model.train(True)  
         
          if lr_scheduler: optimizer = lr_scheduler(optimizer, epoch)              
          running_loss, running_corrects = 0.0, 0.0              

          batch = 0          
          for data in dset_loaders[phase]:
              inputs, labels = data              
              optimizer.zero_grad()            

              if LSTM:                                      
                  hidden = model.init_hidden(len(inputs.to(dev)))                        
                  preds = model(inputs.to(dev), hidden)                      
              else:  
                  preds = model(inputs.to(dev))

              loss = criterion(preds, labels.to(dev))                  
              loss.backward()
              optimizer.step()      
                                   
              # store batch performance
              with torch.no_grad():                      
                  running_loss += float(loss.item())        
                  # loss_value.detach()
                  running_corrects += torch.sum(preds.data.max(1)[1] == labels.to(dev).data)                  
                  ytrue_labels.append(labels.data.cpu().detach().numpy())
                  ypred_labels.append(preds.data.max(1)[1].cpu().detach().numpy())                    
                  batch += 1
                  del loss
                 
          with torch.no_grad():                      
              epoch_loss = running_loss / dset_sizes[phase]
              epoch_acc = running_corrects.cpu().numpy()/dset_sizes[phase]        
             
              train_losses.append(epoch_loss)
              train_accs.append(epoch_acc)                  
              train_labels.append(dict(ypred = ypred_labels, ytrue = ytrue_labels))                      
             
              if verbose > 1: print('{} loss: {:.4f}, acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                 
          # Validation phase
          phase = 'val'
          model.train(False)            
         
          running_loss, running_corrects = 0.0, 0.0              
          ypred_labels, ytrue_labels = [], []
         
          for data in dset_loaders[phase]:
              inputs, labels = data              
              # do not store gradients
              with torch.no_grad():
                  if LSTM:                                      
                      hidden = model.init_hidden(len(inputs.to(dev)))                        
                      preds = model(inputs.to(dev), hidden)                      
                  else:  
                      preds = model(inputs.to(dev))  
                     
                  loss = criterion(preds, labels.to(dev))                            
                 
                  running_loss     += float(loss.item())                  
                  running_corrects += torch.sum(preds.data.max(1)[1] == labels.to(dev).data)
                     
                  ytrue_labels.append(labels.data.cpu().detach().numpy())
                  ypred_labels.append(preds.data.max(1)[1].cpu().detach().numpy())                    
 
          with torch.no_grad():
                             
              val_loss = running_loss / dset_sizes[phase]
              val_acc = running_corrects.cpu().numpy()/dset_sizes[phase]       
           

              val_losses.append(val_loss)
              val_accs.append(val_acc)                
              val_labels.append(dict(ypred = ypred_labels, ytrue = ytrue_labels))
              val_corrects[epoch] = running_corrects

              if verbose > 1: print('{} loss: {:.4f}, acc: {:.4f}'.format(phase, val_loss, val_acc))
              # Deep copy the best model using early stopping
              if val_acc > best_acc:
                  best_acc = val_acc
                  best_epoch = epoch
                  best_model = copy.deepcopy(model)  
       
          early_stopping(val_loss, best_model)
          if early_stopping.early_stop:
              print("Early stopping")
              break
                    
                 
      ########################################              
      time_elapsed = time.time() - start_time  
     
      # ytrue and ypred from the best model during the training            
      def best_epoch_labels(train_labels, best_epoch):        
        for jj in range(len(train_labels[best_epoch]['ypred'])-1):    
            if jj == 0:
              ypred = train_labels[best_epoch]['ypred'][jj]              
              ytrue = train_labels[best_epoch]['ytrue'][jj]                  
            ypred = np.concatenate([ypred, train_labels[best_epoch]['ypred'][jj+1]])
            ytrue = np.concatenate([ytrue, train_labels[best_epoch]['ytrue'][jj+1]])            
        return ypred, ytrue
     
      ytrain_best = best_epoch_labels(train_labels, best_epoch)
      yval_best   = best_epoch_labels(val_labels, best_epoch)      
     
      info = dict(ytrain = ytrain_best, yval = yval_best,
                  best_epoch = best_epoch, best_acc = best_acc)
     
      if verbose > 0:
          print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
          print('Best val Acc: {:4f}'.format(best_acc))
          print('Best Epoch :', best_epoch+1)          
         
      return best_model, train_losses, val_losses, train_accs, val_accs, info, val_corrects
  
    
###########################       
import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            
        elif score <self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
      
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

#%%%
# Define the dataset
filename1 = "TenHealthyData.pickle"

#dataset = 'EPFL'
dataset = 'BNCI'
with open(filename1, 'rb') as fh:
    d1 = pickle.load(fh)
    
    
ss1 = []
for ii in range(len(d1)):
    ss1.append(subject_specific([ii], d1, dataset))
ch_a1 = d1[0].ch_names


batch_size = 32
verbose = 1
augmentdata = dict(std_dev = 0.01,  multiple = None)
num_epochs = 250
learning_rate = 1e-4 # 1e-3
weight_decay  = 1e-5 
verbose = 2 


fs = 80
crop_length = 1.5
crop = dict(fs = fs, crop_length = crop_length)


# define kernel size in terms of ms length 
timE = 100 #ms
ker = timE*fs//1000    

# ker = 8 #timelength//chans         
# convolution parameters 
a, a1 = 3, 1
b, b1 = 3, 3
c, c1 = 3, 5       

     
params = {'conv_channels': [[1, 128,64,32,16] , 
                            [1, 128,64,32,16],
                                [1, 128,64,32,16] ,
                                [1, 64,32,16],
                                [1,32,16], 
                                 [1, 64,32,16],
                                  [1,32,16], 
                                   [1, 64,32,16], [1, 128,64,32,16],
                                  
    ],
                             # ConvDOwn=False
                                                     
                                                                   
					
              'kernel_size':    [[(5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5)],
                                  [(7, 7), (5, 5), (3, 3), (3, 3), (3, 3), (3, 3)],
                                  [(5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5)],
                                  [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3)], 
                                   [(5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5)],
                                    [(7, 7), (5, 5), (3, 3), (3, 3), (3, 3), (3, 3)],
                                    [(7, 7), (7, 7),( 7, 7), (7, 7), (7, 7),( 7, 7)] ,
                                      [(7, 7), (5, 5), (3, 3), (3, 3), (3, 3), (3, 3)],
                                    [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
                                  
                                  ]         }                   
   
        
    
    
keys = list(params)




#### 
input_size = (1, len(ch_a1), ss1[1][0]['xtrain'].shape[2])

'''


table = pd.DataFrame(columns=['Testing_Accuracy', 'AUC', 'TN', 'FP', 'FN', 'TP'])


table_wrt_S = pd.DataFrame(columns=['ACC_C[1, 32, 16]_K[3,[  # ConvDOwn=True
                              [1, 8, 16, 32, 64, 128, 256],
                              [1, 8, 16, 32, 64, 128, 256],
                              [1, 8, 16]]
                             # ConvDOwn=False
                                                     
                                                                   
					
              'kernel_size':    [[(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3)],
              [(7, 7), (5, 5), (3, 3), (3, 3), (3, 3), (3, 3)],
              [(7, 7), (5, 5), (3, 3), (3, 3), (3, 3), (3, 3)]]
                                                                               
              }   3, 3]', 'AUC', 'TN', 'FP', 'FN', 'TP'])
'''
import os
import re
os.getcwd()  ## to know the current directory
directory='72models_BNCI_subj0/subj1'   # type the directory where your best models are saved
#directory='Models_data_BNCI/OneBestModelForEachSubject/Subject0As_a_Test'
#directory='9models'
list_of_models=os.listdir(directory) # create the list of best models
#list_of_models.remove('details_#epoch250.txt')
#list_of_models.remove('details.txt')
preds_saves=[]
count_model=0
validation=[]             
#%%
#for model_from_list in list_of_models:
            
#%%
#import io

################## CHECK the accuracies for validation subject
torch.cuda.empty_cache() 
preds_save=[]
#
preds_saves=[]
y_valid_list=[]
for model_from_list in list_of_models:
#model_from_list=list_of_models[count_model]
    print(count_model)
    
    ### Extract the parameters from the names of the models
    m=re.search('C_(.+?)_', model_from_list)
    k=re.search('_K(.+?)_', model_from_list)
    #k=re.search('_K(.+?)V', model_from_list)
    d=re.search('_D(.+?)_', model_from_list)
    test_sub_val=re.search('S(.+)', model_from_list)
    #test_sub_val=re.search('V_(.+?)', model_from_list)
    if m:
        conv_channels=eval(m.group(1))  
        print(conv_channels) 
    if k:
        kernel_sizes=eval(k.group(1))     
    if d:
        drop_out=eval(d.group(1))
    if test_sub_val:
        test_sub_val=test_sub_val.group(1)
        test_sub_val=test_sub_val[2]
                      
        
    '''
    conv_channels=params['conv_channels'][count_model]
    kernel_sizes=params['kernel_size'][count_model]
    test_sub_val=re.search('SUB_(.+?)', model_from_list)
    count_model=count_model+1
 

    drop_out=0.1
    if test_sub_val:
        test_sub_val=test_sub_val.group(1)

    '''
    
    
    #drop_out=0.1
    
    subj_for_test = eval(test_sub_val)

    
    preds_save=[]
    
    
    
    xtest = subject_specific([subj_for_test], d1, dataset)
    
    x_test, y_test = map(torch.FloatTensor, (xtest[0]['xtrain'], xtest[0]['ytrain']))
    # x_test = torch.flatten(x_test, start_dim = 1)
    x_test = torch.unsqueeze(x_test, dim=1)
    
    # x_test = torch.unsqueeze(x_test, 1)
    print("X test shape", x_test.shape)
    y_test = y_test.type("torch.LongTensor")
    
      
      
    values=(conv_channels,kernel_sizes)
    
    d_t = dict(zip(keys, values))
    
    
    
    
    d_t = dict(zip(keys, values))
    model_to_test = CNN2D(input_size, kernel_size=d_t['kernel_size'],  ##
                              conv_channels=d_t['conv_channels'],
                                                        dense_size    = 256,
                                                        dropout       =drop_out)
    
    
    model_from_list=directory+'/'+model_from_list
    if torch.cuda.is_available():
        model_to_test.load_state_dict(torch.load(model_from_list))    
    else:
        model_to_test.load_state_dict(torch.load(model_from_list, map_location='cpu'))
        
    preds_save=[]
    batch_test = 100;
    n_test_size = x_test.shape[0];
    for ndx in range(0, n_test_size, batch_test):
        with torch.no_grad():
            x_test_batch = x_test[ndx:min(ndx + batch_test, n_test_size)]
            y_test_batch = y_test[ndx:min(ndx + batch_test, n_test_size)]
        
            preds = model_to_test(x_test_batch.to(dev))
            preds_save.append(preds)
    preds_save= torch.cat(preds_save)
    preds_saves.append(preds_save)
    y_valid_list.append(y_test)
    
    ### Upto this point the predictions for validation subject were created 
#%%
    ### Create probabilities for the validation subjects and list of accuracies
preds_classes=[] 
valid_acc_list=[]  
probab_list_valid=[]

sm = torch.nn.Softmax()

for kk in range (0,71):
#for kk in range (0,8):
    preds_class = preds_saves[kk].data.max(1)[1]
    preds_classes.append(preds_class)
    
    
    
        
    corrects = torch.sum(preds_class == y_valid_list[kk].data.to(dev))
    test_acc = corrects.cpu().numpy() / x_test.shape[0]
    print("Test Accuracy :", test_acc)
    valid_acc_list.append(test_acc)
    
    probabilities=sm(preds_saves[kk])
    probab_list_valid.append(probabilities)
    
##################### TESTING!!!! ####################
torch.cuda.empty_cache() 
preds_save=[]
#
preds_saves=[]
y_test_list=[]
for model_from_list in list_of_models:
#model_from_list=list_of_models[count_model]

    
    
    print(count_model)
    
    
    m=re.search('C_(.+?)_', model_from_list)
    k=re.search('_K(.+?)_', model_from_list)
    #k=re.search('_K(.+?)V', model_from_list)
    d=re.search('_D(.+?)_', model_from_list)
    test_sub_val=re.search('S(.+)', model_from_list)
    test_sub_val=re.search('V_(.+?)', model_from_list)
    if m:
        conv_channels=eval(m.group(1))  
        print(conv_channels) 
    if k:
        kernel_sizes=eval(k.group(1))     
    if d:
        drop_out=eval(d.group(1))
    if test_sub_val:
        test_sub_val=test_sub_val.group(1)
        test_sub_val=test_sub_val[2]
                      
        
    '''
    conv_channels=params['conv_channels'][count_model]
    kernel_sizes=params['kernel_size'][count_model]
    test_sub_val=re.search('SUB_(.+?)', model_from_list)
    count_model=count_model+1
 

    drop_out=0.1
    if test_sub_val:
        test_sub_val=test_sub_val.group(1)

    '''
    
    
    #drop_out=0.1
    
    
    
    
    subj_for_test = 1
    
    preds_save=[]
    
    
    
    xtest = subject_specific([subj_for_test], d1, dataset)
    
    x_test, y_test = map(torch.FloatTensor, (xtest[0]['xtrain'], xtest[0]['ytrain']))
    # x_test = torch.flatten(x_test, start_dim = 1)
    x_test = torch.unsqueeze(x_test, dim=1)
    
    # x_test = torch.unsqueeze(x_test, 1)
    print("X test shape", x_test.shape)
    y_test = y_test.type("torch.LongTensor")
    
    
    
    
    
    
    
    values=(conv_channels,kernel_sizes)
    
    d_t = dict(zip(keys, values))
    
    
    
    
    d_t = dict(zip(keys, values))
    model_to_test = CNN2D(input_size, kernel_size=d_t['kernel_size'],  ##
                              conv_channels=d_t['conv_channels'],
                                                        dense_size    = 256,
                                                        dropout       =drop_out)
    
    
    model_from_list=directory+'/'+model_from_list
    if torch.cuda.is_available():
        model_to_test.load_state_dict(torch.load(model_from_list))    
    else:
        model_to_test.load_state_dict(torch.load(model_from_list, map_location='cpu'))
        
    preds_save=[]
    batch_test = 100;
    n_test_size = x_test.shape[0];
    
    ## Create predictions for test subjects 
    for ndx in range(0, n_test_size, batch_test):
        with torch.no_grad():
            x_test_batch = x_test[ndx:min(ndx + batch_test, n_test_size)]
            y_test_batch = y_test[ndx:min(ndx + batch_test, n_test_size)]
        
            preds = model_to_test(x_test_batch.to(dev))
            preds_save.append(preds)
            
    preds_save= torch.cat(preds_save)
    preds_saves.append(preds_save)
    y_test_list.append(y_test)

#%%%
preds_classes=[] 
test_acc_list=[]  
probab_list=[]


## create probabilities and find accuracies for each model using test subjects
sm = torch.nn.Softmax()
for kk in range (0,72):
#for kk in range (0,8):
    preds_class = preds_saves[kk].data.max(1)[1]
    preds_classes.append(preds_class)         
        
    corrects = torch.sum(preds_class == y_test_list[kk].data.to(dev))
    test_acc = corrects.cpu().numpy() / x_test.shape[0]
    print("Test Accuracy :", test_acc)
    test_acc_list.append(test_acc)
    
    probabilities=sm(preds_saves[kk])
    probab_list.append(probabilities)
    
    
 
#%%    
    
    ### Probabilistic classifier ensemble weighting scheme
summ=0
alpha=5
for y in range(0,72):
#for y in range (0,8):
    u=np.float_power(valid_acc_list[y], alpha)*probab_list[y]
    summ=summ+u
    
    
   
w=summ.data.max(1)[1]   

### Result of probablity based scheme
corrects = torch.sum(w == y_test.data.to(dev))
test_acc = corrects.cpu().numpy() / x_test.shape[0]
print(test_acc)


#%%%
### majoruty vote scheme
preds_class_for_majority_vote=[]
for u in range (0, 72):
#for u in range (0,8):

    preds_class_maj = torch.where(preds_classes[u] == 0, torch.tensor(-1), preds_classes[u]) 
    preds_class_for_majority_vote.append(preds_class_maj)

    
majority= sum(preds_class_for_majority_vote)    
    
pred_output_maj = torch.where(majority< 0, torch.tensor(0), majority)
pred_output_maj = torch.where(pred_output_maj > 0, torch.tensor(1), pred_output_maj)
corrects = torch.sum(pred_output_maj == y_test.data.to(dev))
test_acc_maj = corrects.cpu().numpy() / x_test.shape[0]
    
print(test_acc_maj)
    
#%%% weighted Average scheme
weights_accuracies=[]
predictions_from_each_model=[]
valid_acc_list_x=torch.FloatTensor(valid_acc_list)

y=0
for w in range(0, 72):
#for w in range (0,8):
    weight=valid_acc_list[w]/sum(valid_acc_list)
    weights_accuracies.append(weight)
    

    y=y+weight*preds_classes[w].type('torch.FloatTensor')
y=y.to(dev)   
t = Variable(torch.Tensor([0.5]))
y_out_predict=torch.where(y > 0.5, torch.tensor(1).type('torch.FloatTensor').to(dev), y)   
y_out_predict=torch.where(y_out_predict < 0.5, torch.tensor(0).type('torch.FloatTensor').to(dev), y_out_predict)   

corrects = torch.sum(y_out_predict.type('torch.LongTensor').to(dev) == y_test.data.to(dev))
test_acc_weighted = corrects.cpu().numpy() / x_test.shape[0]

print(test_acc_weighted)
#%%%



