import time
import copy
import torch

from nu_smrutils import loaddat
import pandas as pd
# In[2]:
dname = dict(BNCI2014004 = 'aBNCI2014004R.pickle',
             BNCI2014001 = 'aBNCI2014001R.pickle',
             Weibo2014   = 'aWeibo2014R.pickle',
             Physionet   = 'aPhysionetRR.pickle')

# In[3]:
# itemname is one of : ['BNCI2014004', 'BNCI2014001', 'Weibo2014', 'Physionet']
itemname = 'BNCI2014004'
filename = dname[itemname]
iname = itemname + '__'    

# ### Load pooled data
# In[4]:
from nu_smrutils import load_pooled, augment_dataset, crop_data
# In[5]:
from nu_smrutils import loaddat
# In[6]:
data = loaddat(filename)

# In[7]:
subjectIndex = list(range(108))
class_name = ['left_hand', 'right_hand']

dat = load_pooled(data, subjectIndex, class_name, 
                  normalize = True, test_size = 0.15)

# In[8]:
print(dat.keys())
dat['xtrain'].shape

# ### Pytorch dataloaders 
# In[9]:
import torch 
from torch.utils.data import TensorDataset, DataLoader  

#%%
def get_data_loaders(dat, batch_size, EEGNET = None):    
    # convert data dimensions to into to gray scale image format
    if EEGNET: ### EEGNet model requires the last dimension to be 1 
        ff = lambda dat: torch.unsqueeze(dat, dim = -1)    
    else:
        ff = lambda dat: torch.unsqueeze(dat, dim = 1)    
    
    x_train, x_valid, x_test = map(ff,(dat['xtrain'], dat['xvalid'],dat['xtest']))    
    y_train, y_valid, y_test = dat['ytrain'], dat['yvalid'], dat['ytest']
    print('Input data shape', x_train.shape)       
    
    # TensorDataset & Dataloader    
    train_dat    = TensorDataset(x_train, y_train) 
    val_dat      = TensorDataset(x_valid, y_valid) 
    
    train_loader = DataLoader(train_dat, batch_size = batch_size, shuffle = True)
    val_loader   = DataLoader(val_dat,   batch_size = batch_size, shuffle = False)

    output = dict(dset_loaders = {'train': train_loader, 'val': val_loader}, 
                  dset_sizes  =  {'train': len(x_train), 'val': len(x_valid)},
                  test_data   =  {'x_test' : x_test, 'y_test' : y_test})          
    return output 


# In[10]:
dat = get_data_loaders(dat, batch_size = 64)
dat.keys()

# In[11]:
dset_loaders = dat['dset_loaders']
dset_sizes   = dat['dset_sizes']
dset_sizes

dtrain = dset_loaders['train']
dval   = dset_loaders['val']

dtr = iter(dtrain)
dv  = iter(dval)

#% get input size (channel x timepoints)
input_size = (1, dat['test_data']['x_test'].shape[-2], 
                 dat['test_data']['x_test'].shape[-1])
print(input_size)

# In[12]:
inputs, labels = next(dtr)
print(inputs.shape, labels.shape) 

# ## NNet Hyperparameters 
# In[14]:
import torch 
from nu_smrutils import CNN2D

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if dev.type == 'cuda':
   print('Your GPU device name :', torch.cuda.get_device_name())        
else:
   print('No GPU Available :', dev)

# In[15]:
num_epochs = 1 
learning_rate = 1e-3
weight_decay = 1e-4  
batch_size = 64
verbose = 2

fs = 80
# define kernel size in terms of ms length 
time_window = 100 #ms
width = time_window*fs//1000    

# ker = 8 #timelength//chans 
h, w = 3, 1  #hight and width of a rectangular kernel      

kernel_size = [(h, w*width), (h, w*width), (h, w*width),
               (h, w*width), (h, w*width), (h, w*width)]

conv_chan   = [1, 64, 32, 16, 8]            

# In[16]:
kernel_size

# In[17]:
# Define the architecture
model = CNN2D(input_size   = input_size, 
              kernel_size  = kernel_size, 
              conv_channels= conv_chan,
              dense_size   = 256, 
              dropout      = 0.5)               
# In[18]:
# optimizer and the loss function definition 
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, 
                             weight_decay = weight_decay)
criterion = torch.nn.CrossEntropyLoss()

model.to(dev)  
criterion.to(dev)       

print("Model architecture >>>", model)
# ### Training Loop
# In[19]:
import time 
start_time = time.time()          

best_model, best_acc = model, 0.0 
train_losses, val_losses, train_accs = [],[],[]
val_accs, train_labels, val_labels   = [],[],[]

###############################################
# In[20]:
for epoch in range(num_epochs):     
  if verbose > 1: print('Epoch {}/{}'.format(epoch+1, num_epochs))          
  #### TRAIN PHASE ####
  ypred_labels, ytrue_labels = [], []   
  model.train(True)      
  running_loss, running_corrects = 0.0, 0.0
  
  #### loop across batches ####
  for batch, datum in enumerate(dset_loaders['train']):          
      # first batch etc 
      if not batch % 20:
         print('Processing batch: {} / Data : {}:'.format(batch, inputs.shape))
            
      inputs, labels = datum # get data 
      optimizer.zero_grad() # zero gradient         
    
      preds= model(inputs.to(dev)) # make the prediction  
        
      loss = criterion(preds, labels.type(torch.LongTensor).to(dev)) # Calculate loss          
      loss.backward() # Backpropogate       
      optimizer.step()# Update the weights          
        
      with torch.no_grad():
          # storch batch training performance 
          running_loss     += float(loss.item())                  
          running_corrects += torch.sum(preds.data.max(1)[1] == labels.data.to(dev))           
          ytrue_labels.append(labels.data.cpu().detach().numpy())
          ypred_labels.append(preds.data.max(1)[1].cpu().detach().numpy())        
          del loss  
        
  # ********* get the epoch loss and accuracy *********         
  epoch_loss = running_loss / dset_sizes['train']
  epoch_acc  = running_corrects.cpu().numpy()/dset_sizes['train']     

  train_losses.append(epoch_loss)
  train_accs.append(epoch_acc) 
  train_labels.append(dict(ypred = ypred_labels, ytrue = ytrue_labels))                  

  # VALIDATION PHASE #
  model.train(False)     
  for batch, vdatum in enumerate(dset_loaders['val']): 
      inputs, labels = vdatum
      if not batch % 10:
        print('Validate batch: {} / Data : {}:'.format(batch, inputs.shape))       

      preds = model(inputs.to(dev))  # predict and calculate the loss         
      loss  = criterion(preds, labels.type(torch.LongTensor).to(dev))               
      
      running_loss     += float(loss.item())                  
      running_corrects += torch.sum(preds.data.max(1)[1] == labels.data.to(dev))           
      ytrue_labels.append(labels.data.cpu().detach().numpy())
      ypred_labels.append(preds.data.max(1)[1].cpu().detach().numpy())              
      del loss     
  # ********* get the epoch loss and accuracy *********         
  vepoch_loss = running_loss / dset_sizes['val'] 
  vepoch_acc  = running_corrects.cpu().numpy()/dset_sizes['val']  

  # select the model based on the vaidation accuracy 
  if  vepoch_acc > best_acc:
      best_acc, best_epoch = vepoch_acc, epoch 
      best_model = copy.deepcopy(model)           

  # save the results  
  val_losses.append(vepoch_loss)
  val_accs.append(vepoch_acc) 
  val_labels.append(dict(ypred = ypred_labels, ytrue = ytrue_labels))                  
  # ***************************************************        
# In[25]:
def get_best_epoch_labels(train_labels, best_epoch):
    """This function is used by train_model to store best epoch labels"""
    import numpy as np

    for jj in range(len(train_labels[best_epoch]['ypred'])-1):    
        if jj == 0: 
          ypred = train_labels[best_epoch]['ypred'][jj]              
          ytrue = train_labels[best_epoch]['ytrue'][jj] 
            
        ypred = np.concatenate([ypred, train_labels[best_epoch]['ypred'][jj+1]])
        ytrue = np.concatenate([ytrue, train_labels[best_epoch]['ytrue'][jj+1]])            
    return ypred, ytrue
# In[27]:
time_elapsed = time.time() - start_time  

# ytrue and ypred from the best model during the training    
ytrain_best = get_best_epoch_labels(train_labels, best_epoch)
yval_best   = get_best_epoch_labels(val_labels,   best_epoch)      

info = dict(ytrain = ytrain_best, yval = yval_best, 
          best_epoch = best_epoch, best_acc = best_acc)

if verbose > 0:
  print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, 
                                                      time_elapsed % 60))
  print('Best val Acc: {:4f}'.format(best_acc)) 
  print('Best Epoch :', best_epoch+1) 

#return best_model, train_losses, val_losses, train_accs, val_accs, info
####################################################################
