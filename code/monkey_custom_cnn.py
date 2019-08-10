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






import torch 
#import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
#from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


torch.set_default_tensor_type('torch.FloatTensor')



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
                               transforms.Normalize(mean=[0., 0., 0.],
                                                    std=[1., 1., 1.])
                             ]))
train_loader = torch.utils.data.DataLoader(training_set, batch_size = 64, shuffle = True)

validation_set = MonkeyDataset(X_test,y_test, transform=transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0., 0., 0.],
                                                    std=[1., 1., 1.])
                             ]))
test_loader = torch.utils.data.DataLoader(validation_set, batch_size = 64, shuffle = False)



torch.set_default_tensor_type('torch.DoubleTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_epochs = 200
batch_size_train = 64
batch_size_test = 64
learning_rate = 0.001
momentum = 0.9
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4_drop = nn.Dropout2d()
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv5_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(7*7*128, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv3(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv4_drop(self.conv4(x)), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv5_drop(self.conv5(x)), kernel_size=2, stride=2))
        x = x.view(-1,7*7*128 )
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    
    
    
network = Net().to(device).float()

optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=0.01)




train_losses = []
train_counter = []
test_losses = []
test_counter = [i for i in range(n_epochs + 1)]


def train(epoch):
  network.train()
  for batch_idx, (Data, target) in enumerate(train_loader):
    Data = Data.to(device, dtype=torch.float)
    target = target.to(device).long()
    optimizer.zero_grad()
    output = network(Data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(Data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(epoch-1)

      
      
def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for Data, target in test_loader:
      Data = Data.to(device, dtype=torch.float)
      target = target.to(device).long()
      output = network(Data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
  
  
  
test()
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()
  
print('done')
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.plot(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('epochs')
plt.ylabel('negative log likelihood loss')
fig





