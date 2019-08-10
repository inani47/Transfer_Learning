# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 19:00:28 2018

@author: PI
"""
import pandas as pd
import os
import numpy as np
import matplotlib.image as mpimg 

from skimage import io
import skimage
from skimage.transform import resize as transf


from torch.optim import lr_scheduler
import torchvision
import time
import copy




import torch 
#import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
#from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models


torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



cols = ['Label','Latin Name', 'Common Name','Train Images', 'Validation Images']
labels = pd.read_csv("10-monkey-species/monkey_labels.txt", names=cols, skiprows=1)
labels = labels['Common Name']
print(labels)
train_dir = "10-monkey-species/training/training/"
test_dir =  "10-monkey-species/validation/validation/"



def get_data(folder):
    """
    Load the data and labels from the given folder.
    """
    X = []
    y = []
    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            if folderName in ['n0']:
                label = 0
            elif folderName in ['n1']:
                label = 1
            elif folderName in ['n2']:
                label = 2
            elif folderName in ['n3']:
                label = 3
            elif folderName in ['n4']:
                label = 4
            elif folderName in ['n5']:
                label = 5
            elif folderName in ['n6']:
                label = 6
            elif folderName in ['n7']:
                label = 7
            elif folderName in ['n8']:
                label = 8
            elif folderName in ['n9']:
                label = 9
            else:
                label = 10
            for image_filename in os.listdir(folder + folderName):
                img_file = mpimg.imread(folder + folderName + '/' + image_filename)
                if img_file is not None:
                    img_file = transf(img_file, (224, 224, 3))
                    
                    #img_file = scipy.misc.imresize(arr=img_file, size=(150, 150, 3))
                    img_arr = np.asarray(img_file)
#                    img_arr = np.transpose(img_arr, (2,0,1))
                    X.append(img_arr)
                    y.append(label)
    X = np.asarray(X)
    y = np.asarray(y)
    return X,y
X_train, y_train = get_data(train_dir)
X_test, y_test= get_data(test_dir)






class MonkeyDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, images, labels, transform=None):
        'Initialization'
        self.labels = labels
        self.images = images
        self.transform = transform

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.images)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        # Load data and get label
        X = self.images[index]
        y = self.labels[index]
        
        if self.transform:
            X = self.transform(X)

        return X, y
    
    
    
params = {'batch_size': 64,
          'shuffle': True}
  
training_set = MonkeyDataset(X_train,y_train,transform=transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                             ]))
train_loader = torch.utils.data.DataLoader(training_set, batch_size = 64, shuffle = True)

validation_set = MonkeyDataset(X_test,y_test, transform=transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                             ]))
test_loader = torch.utils.data.DataLoader(validation_set, batch_size = 64, shuffle = True)





def train_model(model, criterion, optimizer, scheduler=None, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    counter = [i for i in range(num_epochs)]
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                dat = train_loader
#                scheduler.step()
                model.train()  # Set model to training mode
            else:
                dat = test_loader
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dat:
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device)
                labels = labels.long()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / (len(dat.dataset))
            epoch_acc = running_corrects.double() / (len(dat.dataset))
            
            if phase == 'train':
                train_losses.append(epoch_loss)
            else:
                test_losses.append(epoch_loss)
                
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))


     #plot loss vs epoch
    fig = plt.figure()
    plt.plot(counter, train_losses, color='blue')
    plt.plot(counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('epochs')
    plt.ylabel('Cross Entropy Loss')
    fig
    fig.savefig('Transfer_feat_extact.png')
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model_conv = models.resnet18()
model_conv.load_state_dict(torch.load('resnet18-5c106cde.pth'))

#Freeze layers
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 10)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=14, gamma=0.1)

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=14)