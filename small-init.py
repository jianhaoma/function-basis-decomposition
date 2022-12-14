# imports
import wandb
import flash
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import copy
from IPython.display import clear_output
from typing import List
from flash.core.optimizers import LARS
from flash.core.optimizers import LAMB
import os
import argparse
from nngeometry.object.fspace import FMatDense
from nngeometry.object.vector import FVector
from nngeometry.object import PMatImplicit
from nngeometry.generator import Jacobian
from nngeometry.layercollection import LayerCollection

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import copy
from IPython.display import clear_output 
from sklearn.utils.extmath import randomized_svd
from nngeometry.object.fspace import FMatDense
from nngeometry.object.vector import FVector
from nngeometry.object import PMatImplicit
from nngeometry.generator import Jacobian
from nngeometry.layercollection import LayerCollection
clear_output()
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# !! fixed random seed! 

# Download training dataset
dataset = MNIST(root='data/', download=True)
vali_dataset = MNIST(root='data/', train=False)

# Transform to tensors
train_dataset = MNIST(root='data/', 
                train=True,
                transform = transforms.ToTensor())

# training data and validation data.
train_ds = train_dataset
val_ds = MNIST(root='data/', 
                train=False,
                transform = transforms.ToTensor())

# Specify device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build CNN models, 3 convolution layers.
def conv_block(input_channels, out_channels, kernel_size, padding, batch_norm = True, pool=True):
    layers = [nn.Conv2d(input_channels, out_channels, kernel_size, 1, padding)]
    if batch_norm: 
      layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

def in_out_channel(in_channel=128, layer=1, rate=1/2, mode='exp'):
  if mode == 'exp':
    out_channel = int(in_channel*rate**layer)
  if mode == 'linear':
    out_channel = int(in_channel - layer*rate)
  return out_channel
def classifier_in_dim(out):
  return len(torch.flatten(out, start_dim=1))

class CNN3(nn.Module):
    def __init__(self, num_conv_layers = 3, input_channels=1, out_channels_base=16, rate=2, mode = 'exp',
                  num_classes=10, batch_norm = True, pool = True):
        super().__init__()
        self.conv1 = conv_block(
            input_channels, out_channels_base, 
            kernel_size=3, padding=1, batch_norm=batch_norm, pool=pool)
        self.conv2 = conv_block(
            out_channels_base, 
            in_out_channel(out_channels_base, layer=1, rate=rate, mode=mode), 
            kernel_size=3, padding=1, batch_norm=batch_norm, pool=pool)
        self.conv3 = conv_block(
            in_out_channel(out_channels_base, layer=1, rate=rate, mode=mode), 
            in_out_channel(out_channels_base, layer=2, rate=rate, mode=mode), 
            kernel_size=3, padding=1, batch_norm=batch_norm, pool=pool)
        self.lin_batch = nn.BatchNorm1d(288)
        self.classifier = nn.Linear(288, 10, bias=False)  # No bias. No softmax

    def forward(self, out):
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = torch.flatten(out, start_dim=1)
        out = self.lin_batch(out)
        out = self.classifier(out)
        return out

# Define Data Loaders
_CONV_OPTIONS = {"kernel_size": 3, "padding": 1, "stride": 1}
def get_activation(activation: str):
    if activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'hardtanh':
        return torch.nn.Hardtanh()
    elif activation == 'leaky_relu':
        return torch.nn.LeakyReLU()
    elif activation == 'selu':
        return torch.nn.SELU()
    elif activation == 'elu':
        return torch.nn.ELU()
    elif activation == "tanh":
        return torch.nn.Tanh()
    elif activation == "softplus":
        return torch.nn.Softplus()
    elif activation == "sigmoid":
        return torch.nn.Sigmoid()
    else:
        raise NotImplementedError("unknown activation function: {}".format(activation))

def get_pooling(pooling: str):
    if pooling == 'max':
        return torch.nn.MaxPool2d((2, 2))
    elif pooling == 'average':
        return torch.nn.AvgPool2d((2, 2))
    elif pooling == 'id':
        return torch.nn.Identity()

def fully_connected_net(num_chann: int, num_classes: int, pic_size: int, widths: List[int], activation: str, bias: bool = True) -> nn.Module:
    modules = [nn.Flatten()]
    for l in range(len(widths)):
        prev_width = widths[l - 1] if l > 0 else num_chann*pic_size**2
        modules.extend([
            nn.Linear(prev_width, widths[l], bias=bias),
            get_activation(activation),
        ])
    modules.append(nn.Linear(widths[-1], num_classes, bias=False))
    return nn.Sequential(*modules)

def fully_connected_net_bn(num_chann: int, num_classes: int, pic_size: int, widths: List[int], activation: str, bias: bool = True) -> nn.Module:
    modules = [nn.Flatten()]
    for l in range(len(widths)):
        prev_width = widths[l - 1] if l > 0 else num_chann*pic_size**2
        modules.extend([
            nn.Linear(prev_width, widths[l], bias=bias),
            get_activation(activation),
            nn.BatchNorm1d(widths[l])
        ])
    modules.append(nn.Linear(widths[-1], num_classes, bias=False))
    return nn.Sequential(*modules)

def convnet(num_chann: int, num_classes: int, pic_size: int, widths: List[int], activation: str, pooling: str, bias: bool) -> nn.Module:
    modules = []
    size = pic_size
    for l in range(len(widths)):
        prev_width = widths[l - 1] if l > 0 else num_chann
        if l ==0 :
            modules.extend([
                nn.Conv2d(prev_width, widths[l], bias=bias, **_CONV_OPTIONS),
                get_activation(activation),
                get_pooling('id'),
            ])
        else:
            modules.extend([
                nn.Conv2d(prev_width, widths[l], bias=bias, **_CONV_OPTIONS),
                get_activation(activation),
                get_pooling(pooling),
            ])
    size //= 2
    modules.append(nn.Flatten())
    modules.append(nn.Linear(widths[-1]*size**2, num_classes, bias=False))
    return nn.Sequential(*modules)

def convnet_bn(num_chann: int, num_classes: int, pic_size: int, widths: List[int], activation: str, pooling: str, bias: bool) -> nn.Module:
    modules = []
    size = pic_size
    for l in range(len(widths)):
        prev_width = widths[l - 1] if l > 0 else num_chann
        if l == 0 :
            modules.extend([
                nn.Conv2d(prev_width, widths[l], bias=bias, **_CONV_OPTIONS),
                get_activation(activation),
                get_pooling('id'),
            ])
        else:
            modules.extend([
            nn.Conv2d(prev_width, widths[l], bias=bias, **_CONV_OPTIONS),
            get_activation(activation),
            nn.BatchNorm2d(widths[l]),
            get_pooling(pooling),
        ])
    size //= 2
    modules.append(nn.Flatten())
    modules.append(nn.Linear(widths[-1]*size*size, num_classes, bias=False))
    return nn.Sequential(*modules)

# model = convnet(1, 10, 28, [28, 28], activation='relu', pooling='average', bias=True)

# Define a hook
features = {}
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

# Train the model
def vali_step(model, device, val_loader, compute_phi=False, gradient=False):
    model.eval()
    outputs = []
    FEATS = []
    num_batch = len(val_loader)
    f_out = torch.empty((0, 10))
    vacc = 0
    total = 0
    correct = 0
    grad_norm = 0
    if gradient:
      lc = LayerCollection.from_model(model)
      Jac = Jacobian(model, n_output=10, centering=True, layer_collection=lc)
    for batch in val_loader:
      images, labels = batch
      images, labels = images.to(device), labels.to(device)
      output = model(images)
      total += labels.size(0)
      _, predicted = torch.max(output.data, 1)
      correct += (predicted == labels).sum().item()
      labels=F.one_hot(labels.to(torch.int64), 10).float()    # !! number of labels used here!!              
      loss = F.mse_loss(output, labels) # Calculate loss, mean suqare loss
      f_out = torch.vstack((f_out, output.detach().cpu()))
      if compute_phi:
        FEATS.append(features['phi'].cpu().numpy())
      if gradient:
        grad_norm += torch.norm((Jac.get_jacobian([images, labels])).detach().cpu(), 'fro')**2
    vacc = correct/total
    f_norm = torch.norm(output.detach())
    if compute_phi:
      phi = torch.flatten(torch.from_numpy(FEATS[0]), 1)
      for i in range(num_batch-1):
        phi = torch.vstack((phi, torch.flatten(torch.from_numpy(FEATS[i+1]), 1)))
    else:
      phi = None
    return vacc, phi, output.detach(), f_out, np.sqrt(grad_norm)

def train(epochs, max_lr, decrease_lr, model, scheduler, device, train_loader, val_loader, first_kernel_base, rate, 
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    Iter = len(train_loader)
    history = []
    Phi = {}
    F_out = {}
    train_acc = []
    val_acc = []
    lr = max_lr
    train_loss = []
    lrs = []
    valtime = []
    iter = 0
    val_iter = 0
    Grad_norm = []
    F_norm = []
    Hessian_norm = []
    train_batches = len(train_loader)

    model.eval()
    vacc, _, _, f_out, _ = vali_step(model, device, val_loader, compute_phi=False, gradient=False)
    val_acc.append(vacc)
    F_out0 = f_out
    valtime.append(0)
    for epoch in range(epochs):

        # update
        if decrease_lr:
          lr = max_lr - (max_lr-0.01)/int(epochs/10)*int(epoch/10)
        # optimizer
        optimizer = opt_func
        total = 0
        correct = 0
        for batch in train_loader:
          model.train()
          images, labels = batch 
          images, labels = images.to(device), labels.to(device)
          labels=F.one_hot(labels.to(torch.int64), 10).float()    # !! number of labels used here!!              
          out = model(images)                  # Generate predictions
          loss = F.mse_loss(out, labels)
          loss.backward()
          train_loss.append(loss.detach())
          total += labels.size(0)
          correct += (out == labels).sum().item()
          train_acc.append(correct/total)
          # Gradient clipping
          if grad_clip: 
              nn.utils.clip_grad_value_(model.parameters(), grad_clip)
          optimizer.step()
          optimizer.zero_grad()

        scheduler.step()
    # Register a hook for the last layer
    model.lin_batch.register_forward_hook(get_features('phi'))
    model.eval()
    with torch.no_grad():
        vacc, phi, _, f_out, _ = vali_step(model, device, val_loader, compute_phi=True)
        beta = model.state_dict()['classifier.weight'].detach().cpu()
        F_out1 = f_out
    return train_loss, train_acc, val_acc, phi, beta, F_out, valtime, F_out0, F_out1

out_channels_base=128
rate=1/2
mode = 'exp'

# experiment setting: depth of net
num_conv_layers = 3
batch_size = 512
test_size = batch_size
batch_norm = False
epochs = 10 #9
max_lr = 2 #0.05
decrease_lr = False
grad_clip = 0.1
# weight_decay = 1e-4
step_size = 50
weight_decay = 0.33
train_loader = DataLoader(train_ds, batch_size, shuffle=True) 
val_loader = DataLoader(val_ds, test_size)

model = CNN3(num_conv_layers = num_conv_layers,
    input_channels=1, out_channels_base=out_channels_base, 
    rate=rate, mode = mode,
    num_classes=10, batch_norm = batch_norm, pool = True).to(device)

opt_func = LARS(model.parameters(), lr=max_lr, momentum=0.9, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.StepLR(opt_func, step_size, gamma=weight_decay)

train_loss, train_acc, val_acc, phi, beta, F_out, valtime, F_out0, F_out1 = train(epochs, max_lr, decrease_lr, model, scheduler, device, train_loader, val_loader, out_channels_base, rate, weight_decay, grad_clip, opt_func)

train_loss = [x.detach().cpu().numpy() for x in train_loss]

from datetime import datetime
from pytz import timezone     

Ue = timezone('US/Eastern')
Ue_time = datetime.now(Ue)
time = Ue_time.strftime('%m-%d-%H-%M')
path = 'mnist-cnn'+time+'-'+str(seed)
os.makedirs(path)

U, S, V = torch.svd(phi)
beta_star = torch.matmul(beta, V)
A = torch.matmul(V, torch.diag(S))
beta_val = torch.matmul(beta, A)/100

u, s, v = torch.svd(beta_val)
m, n = beta_val.shape
_, d = u.shape
spars = torch.zeros((d, d))
spars0 = torch.zeros((d, d))

for i in range(d):
    for j in range(d):
        beta_val1 = torch.matmul(F_out1.t(), U)/100
        beta_val0 = torch.matmul(F_out0.t(), U)/100
        
        spars[i,j] = (torch.mul(beta_val1, u[:, i].reshape((m, 1)) @ v[:, j].reshape((n, 1)).T).sum())
        spars0[i,j] = (torch.mul(beta_val0, u[:, i].reshape((m, 1)) @ v[:, j].reshape((n, 1)).T).sum())
spars0 = spars0.reshape(-1)
spars = spars.reshape(-1)

residul = np.linalg.norm(np.array(spars0))**2
top = sorted(range(len(spars)), key=lambda i: spars[i], reverse=True)[:20]
reindex = []
reindx_x = []
residul = np.linalg.norm(np.array(spars0))**2
for i in range(len(top)):
  reindx_x.append(str(i+1))
  reindex.append(abs(spars0[top[i]]))
  residul = residul - spars0[top[i]]**2
reindex.append(np.sqrt(residul)) 
reindx_x.append('residual')
stick = []
for i in range(10):
  stick.append(2*i)
stick.append(20)
plt.rcParams['text.usetex'] = True
plt.style.use('seaborn-paper')
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 150
pb = plt.bar(list(reindx_x), list(reindex))
pb[-1].set_color('r')
plt.xticks(stick)
axes = plt.gca()
plt.xlabel("index", color='k')
plt.ylabel("$ |\\beta_i(\\theta_0)|$", color='k')
axes = plt.gca()
axes.xaxis.label.set_size(20)
axes.yaxis.label.set_size(20)
plt.xticks(color='k', fontsize=16)
plt.yticks(color='k', fontsize=16)
plt.tight_layout()
plt.savefig(path+'/small-init'+str(seed)+'.pdf')
torch.save(spars, path+'/spars1')
torch.save(spars0, path+'/spars0')
