# normalization function 
import torch
from torch.utils.data import TensorDataset, DataLoader
  
import numpy as np
import pickle 
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin 
from sklearn.model_selection import train_test_split

###############
def loaddat(filename):
    """Loads pickle file
    """
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    return data

###############
class SKStandardScaler(TransformerMixin):    
    """
    This function performs normalization to the EEG data 
    using the sklearn standardscaler module.
        
    Read more in sklearn.preprocessing StandardScaler 
        
    Parameters:
    ----------
    Input : 
        A numpy array of shape (samples, channel, times)       
    Returns: 
        Normalized numpy array of shape (samples, channel, times) 
    ---------        
    """    
    def __init__(self, **kwargs):
        self._scaler = StandardScaler(copy=True, **kwargs)
        self._orig_shape = None

    def fit(self, X, **kwargs):
        X = np.array(X)    
        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        X = self._flatten(X)
        self._scaler.fit(X, **kwargs)
        return self
    
    def transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)        
        X = self._scaler.transform(X, **kwargs)
        X = self._reshape(X)
        return X
    
    def _flatten(self, X):        
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):        
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X    

############### 
def load_pooled(data, subjectIndex, class_name, 
                normalize = True, test_size = 0.15):      
    """
    Creates pooled data from all subject specific EEG dataset.          
    
    Parameters:
    -------------------------
    Input: a python list containing MNE EEG data objects. 
    
    For instance, a list with the following elements:    
    [<Epochs  |   720 events, 'left_hand': 360  'right_hand': 360>,
     <Epochs   |  680 events, 'left_hand': 340, 'right_hand': 340>]
  
    Returns:
    -------------------------
    A dictionary :
        X_train, X_valid, X_test: 
        np.array of shape >>>  (samples, channel, times), 
        
        Data labels: 
        y_train, y_valid, y_test
    -------------------------
    output = dict(xtrain = X_train, xvalid = X_valid, xtest = X_test,
                  ytrain = y_train, yvalid = y_valid, ytest = y_test)
    -------------------------    
    """         
    #% extract positive (+1) and negative (-1) classes      
    pos, neg = [], []    
    
    for ii in subjectIndex:
        try: # get numpy data from the mne object                     
            pos.append(data[ii][class_name[0]].get_data())
            neg.append(data[ii][class_name[1]].get_data())            
        except Exception:
            pass 
                
    # prepare the pooled data and concatenate the data from all subjects 
    s1pos, s1neg = pos[-1], neg[-1]     
    for jj in range(len(pos)-1): # all subject but the last one 
        s1pos = np.concatenate([s1pos, pos[jj]])
        s1neg = np.concatenate([s1neg, neg[jj]])     

    # get the labels and construct data array from all subjects 
    X = np.concatenate([s1pos, s1neg])        
    Y = np.concatenate([np.ones(s1pos.shape[0]), np.zeros(s1neg.shape[0])])        
    
    # normalization 
    if normalize:                                    
        scaler = SKStandardScaler()
        X = scaler.fit_transform(X)         
        
    # split the data using sklearn split function 
    x_rest, x_test, y_rest, y_test =\
        train_test_split(X, Y, test_size=test_size, random_state=42,
                         stratify=Y)

    x_train, x_valid, y_train, y_valid =\
        train_test_split(x_rest, y_rest, test_size=0.2, random_state=42,
                         stratify=y_rest)                   

    # Convert to Pytorch tensors
    X_train, X_valid, X_test = map(torch.FloatTensor,  
                                   (x_train, x_valid, x_test))
    y_train, y_valid, y_test = map(torch.FloatTensor, 
                                   (y_train, y_valid, y_test))    

    return dict(xtrain = X_train, xvalid = X_valid, xtest = X_test,
                ytrain = y_train, yvalid = y_valid, ytest = y_test)


###############
def subject_specific(data, subjectIndex, class_name, 
                     normalize = True, test_size = 0.15):      
    """
    Creates a list of subject-specific EEG data with a  
    [Xtrain, Xvalid, Xtest] from a list of MNE objects.        .          
    
    Parameters:
    -------------------------
    INPUT: a python list containing MNE EEG data objects from several subjects. 
    
    For instance, a list from one subject contains the following MNE objects:    
    [<Epochs  |   720 events, 'left_hand': 360  'right_hand': 360>,
    <Epochs   |   680 events, 'left_hand': 340, 'right_hand': 340>]
  
    Returns:
    -------------------------
    A list of dictionaries (e.g. output):
        
        output[0].keys() >> dict_keys(['xtrain', 'xvalid', 'xtest', 
                                        ytrain', 'yvalid', 'ytest'])
                   
        Data np.array:
        'xtrain', 'xvalid', 'xtest' of shape > (samples, channel, times), 
        
        Data labels: 
        y_train, y_valid, y_test
    -------------------------
    OUTPUT = dict(xtrain = X_train, xvalid = X_valid, xtest = X_test,
                  ytrain = y_train, yvalid = y_valid, ytest = y_test)
    -------------------------    
    """    
    
    #% extract positive (+1) and negative (-1) classes          
    pos, neg = [], []      
    datx = []                            
    if len(subjectIndex) > 1:     
        try:           
            for jj in subjectIndex:                
                print('Loading subject:', jj)   
                dat = data[jj]                                     
                pos.append(dat[class_name[0]].get_data())
                neg.append(dat[class_name[1]].get_data())  
        except Exception as err:
            print(err)
    else:
        print('Loading subject:', subjectIndex[0]+1)  
        dat = data[subjectIndex[0]]
        pos.append(dat[class_name[0]].get_data())
        neg.append(dat[class_name[1]].get_data())  

    # subject specific upsampling 
    for ii in range(len(pos)):              
        X, Y = [], []                  
        X = np.concatenate([pos[ii], neg[ii]])            
        Y = np.concatenate([np.ones(pos[ii].shape[0]), 
                            np.zeros(neg[ii].shape[0])])     
        #% normalization 
        if normalize:
            scaler = SKStandardScaler()
            X = scaler.fit_transform(X)            

        x_rest, x_test, y_rest, y_test =\
            train_test_split(X, Y, test_size = test_size, 
                             random_state=42, stratify = Y)    
        
        x_train, x_valid, y_train, y_valid =\
            train_test_split(x_rest, y_rest, test_size = 0.20, 
                             random_state = 42, stratify = y_rest)                   

       # Convert to Pytorch tensors
        X_train, X_valid, X_test = map(torch.FloatTensor, 
                                       (x_train, x_valid, x_test))
        y_train, y_valid, y_test = map(torch.FloatTensor, 
                                       (y_train, y_valid, y_test)) 

        datx.append(dict(xtrain = X_train, xvalid = X_valid, xtest = X_test,
                        ytrain = y_train, yvalid = y_valid, ytest = y_test)) 
        
    return datx  
 
###############
def augment_dataset(X, Y, std_dev, multiple):
    """
    Augments the size of the dataset by introducing unbiased gaussian noise.
    Resulting dataset is 'multiple' times bigger than original.
    
    Parameters:
    -------------------------
        X (torch.FloatTensor): Input training data
        Y (torch.FloatTensor): Target training data
        
        std_dev (float): Standard deviation of gaussian noise to apply
        multiple (int): Factor by how much the dataset should be bigger
    """
    nX, nY = X.clone(), Y.clone()
    
    for i in range(multiple-1):        
        augmented_input  = X + torch.zeros_like(X).normal_(0, std_dev)
        nX  = torch.cat((nX, augmented_input))
        nY  = torch.cat((nY, Y))        
    return nX, nY


############### 
def crop_data(fs, crop_length, xdata, ylabel, xpercent):   
    """ Crop EEG data along time points with pre-defined time segment,
        and generate multiple cropped segments.
        
        Parameters:
        -----------
        fs          : EEG sampling frequency rate
        crop_length : length of crop time-window in seconds
        xdata       : numpy array of shape (samples, channel, times)
        xpercent    : amount of overlap in percentage  
        
    """    
    xoverlap = crop_length*xpercent/100    
    desired_length = np.int(fs*crop_length)
    overlap = np.int(fs*xoverlap) 
     
    number_splits  = xdata.shape[-1]//desired_length    
    tstart, tstop = 0, desired_length
   
    #% needed to copy multiple times
    t = 3 - crop_length    
    for ii in np.arange(number_splits + t):       
        if ii == 0:
            tstart = tstart    
            tstop  = tstart + desired_length + overlap            
            Xi, Yi = xdata[:,:,tstart:tstop], ylabel 
        else:
            try:                
                tstart = tstart + desired_length  
                tstop  = tstart + desired_length + overlap
                
                Xi = torch.cat([Xi, xdata[:,:,tstart:tstop]])
                Yi = torch.cat([Yi, ylabel])                
            except:
                pass       
    return Xi, Yi  


#############################################
import time
import copy
##############################################
dev = torch.device("cpu")

if torch.cuda.is_available():
    dev = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
#############################################################
def best_epoch_labels(train_labels, best_epoch):
    """This function is used by train_model to store best epoch labels"""
    for jj in range(len(train_labels[best_epoch]['ypred'])-1):    
        if jj == 0: 
          ypred = train_labels[best_epoch]['ypred'][jj]              
          ytrue = train_labels[best_epoch]['ytrue'][jj]                  
        ypred = np.concatenate([ypred, train_labels[best_epoch]['ypred'][jj+1]])
        ytrue = np.concatenate([ytrue, train_labels[best_epoch]['ytrue'][jj+1]])            
    return ypred, ytrue

############################################################
def train_model(model, dset_loaders, dset_sizes, criterion, optimizer, dev,
                lr_scheduler = None, num_epochs = 50, verbose = 2, LSTM = False):    
      """
      Method to train a PyTorch neural network with the given parameters for a
      certain number of epochs. Keeps track of the model yielding the best validation
      accuracy during training and returns that model before potential overfitting
      starts happening. Records and returns training and 
      validation losses and accuracies over all epochs.

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
      train_losses, val_losses, train_accs = [],[],[]
      val_accs, train_labels, val_labels   = [],[],[]
      
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
####################################################################
 

