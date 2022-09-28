
# imports
import flash
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
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

clear_output()
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# fixed random seed

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
# Define a hook
features = {}
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

def vali_phi(model, device, val_loader):
    model.eval()
    FEATS = []
    num_batch = len(val_loader)

    for batch in val_loader:
      images, labels = batch
      images, labels = images.to(device), labels.to(device)
      output = model(images)
      FEATS.append(features['phi'].cpu().numpy())

    phi = torch.flatten(torch.from_numpy(FEATS[0]), 1)
    for i in range(num_batch-1):
      phi = torch.vstack((phi, torch.flatten(torch.from_numpy(FEATS[i+1]), 1)))
    return phi

# Train the model
def train(epochs, max_lr, decrease_lr, model, scheduler, device, train_loader, val_loader, weight_decay=0, grad_clip=None, opt_func=None):
    train_acc = []
    lr = max_lr
    train_loss = []
    scheduler = scheduler
    optimizer = opt_func
    state_dict = model.state_dict()
    opt_init = optimizer.state_dict()
    for epoch in range(epochs):

        # optimizer
        total = 0
        correct = 0
        for batch in train_loader:
          model.train()
          images, labels = batch 
          images, labels = images.to(device), labels.to(device)
          labels=F.one_hot(labels.to(torch.int64), 10).float()    
          out = model(images)                  
          loss = F.mse_loss(out, labels)
          loss.backward()
          train_loss.append(loss.detach().cpu().numpy())
          total += labels.size(0)
          _, predicted = torch.max(out.data, 1)
          #correct += (predicted == labels).sum().item()
          train_acc.append(correct/total)
          # Gradient clipping
          if grad_clip: 
              nn.utils.clip_grad_value_(model.parameters(), grad_clip)
          optimizer.step()
          optimizer.zero_grad()
          scheduler.step()

    model.eval()
    f_out = torch.empty((0, 10))
    for _, data in enumerate(val_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        f_out = torch.vstack((f_out, outputs.detach().cpu()))
        
    # Register a hook for the last layer
    handle = model.conv3.register_forward_hook(get_features('phi'))
    model.eval()
    with torch.no_grad():
        phi  = vali_phi(model, device, val_loader)
        beta = model.state_dict()['classifier.weight'].detach().cpu()
        handle.remove()
    return train_loss, train_acc, phi, beta, state_dict, opt_init, f_out

# experiment setting: width of net
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
val_loader = DataLoader(val_ds, test_size, shuffle=False)

model = CNN3(num_conv_layers = num_conv_layers,
    input_channels=1, out_channels_base=out_channels_base, 
    rate=rate, mode = mode,
    num_classes=10, batch_norm = batch_norm, pool = True).to(device)

opt_func = LARS(model.parameters(), lr=max_lr, momentum=0.9, weight_decay=1e-4)
#opt_func = torch.optim.SGD(model.parameters(), lr=max_lr, momentum=0.9,weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.StepLR(opt_func, step_size, gamma=weight_decay)

train_loss, train_acc, phi, W, model_init, opt_init, f_out = train(epochs, max_lr, decrease_lr, model, scheduler, device, train_loader, val_loader, 
                  weight_decay=weight_decay, grad_clip=grad_clip, opt_func=opt_func)

U, S, V = torch.svd(phi)
beta_star = torch.matmul(W, V)

A = torch.matmul(V, torch.diag(S))
beta_val = torch.matmul(W, A)/100

u, s, v = torch.svd(beta_val)
m, n = beta_val.shape

    # spars = list(torch.norm(beta_val, dim=0))
    # top5 = sorted(range(len(spars)), key=lambda i: spars[i], reverse=True)[:5]

top5 = list(range(5))
Beta = {}
for i in range(5):
    Beta[i] = torch.matmul(u[:, i].reshape((m, 1)) @ v[:, i].reshape((n, 1)).T, U.t()).to(device)

def beta_fn(out, basis):
    proj_v = torch.trace(basis.mm(out))/10000
    return proj_v
# Train the model
def vali_step(model, device, val_loader, Beta, b_size):
    model.eval()
    outputs = []
    FEATS = []
    num_batch = len(val_loader)
    f_out = torch.empty((0, 10))
    vacc = 0
    total = 0
    correct = 0
    grad_norm = 0
    b_size = b_size
    Num_base = 5
    Grad = torch.zeros(Num_base)
    beta_val = torch.zeros(Num_base)

    for num, batch in enumerate(val_loader):
      batch_size = len(batch[0])
      images, labels = batch
      images, labels = images.to(device), labels.to(device)
      output = model(images)
      total += labels.size(0)
      for num_base in range(Num_base):
        beta_Fn = beta_fn(output, Beta[num_base][:, num*b_size:num*b_size+batch_size
                                                  ])
        beta_Fn.backward(retain_graph=True)
        parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
        for p in parameters:
            Grad[num_base] += torch.norm(p.grad.detach().data.cpu())**2
            p.grad = None
        beta_val[num_base] += beta_Fn.item()
        del beta_Fn, parameters
        torch.cuda.empty_cache()
    Grad = torch.sqrt(Grad)
    vacc = correct/total
    return vacc, torch.unsqueeze(Grad, 0), torch.unsqueeze(beta_val, 0)

def train_val(epochs, max_lr, model, Beta, scheduler, device, train_loader, val_loader,
                  weight_decay=0, grad_clip=None, opt_func=None, b_size = None):
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
    optimizer = opt_func
    scheduler = scheduler
    Coe_ten = None
    GRAD = None
    BETA = None
    for epoch in range(epochs):

        # optimizer
        total = 0
        correct = 0
        for batch in train_loader:

          # Validation phase
          if iter<400:
                # para = optimizer.param_groups[0]['params']
                vacc, Grad, beta_val = vali_step(model, device, val_loader, Beta, b_size)
                if GRAD == None:
                  GRAD = Grad; BETA = beta_val
                else:
                  GRAD = torch.cat((GRAD, Grad), dim=0)
                  BETA = torch.cat((BETA, beta_val), dim=0)

                val_acc.append(vacc)
                val_acc.append(vacc)
                valtime.append(iter+1)
                
          iter += 1

          model.train()
          images, labels = batch 
          images, labels = images.to(device), labels.to(device)
          labels=F.one_hot(labels.to(torch.int64), 10).float()    
          out = model(images)                  
          loss = F.mse_loss(out, labels)
          loss.backward()
          train_loss.append(loss.detach().cpu().numpy())
          total += labels.size(0)
          _, predicted = torch.max(out.data, 1)
          #correct += (predicted == labels).sum().item()
          train_acc.append(correct/total)
          # Gradient clipping
          if grad_clip: 
              nn.utils.clip_grad_value_(model.parameters(), grad_clip)
          optimizer.step()
          optimizer.zero_grad()
          scheduler.step()
    return train_loss, train_acc, val_acc, F_out, valtime, F_norm, Grad_norm, GRAD, BETA



# model.load_state_dict(model_init)
# opt_func.load_state_dict(opt_init)

# scheduler = torch.optim.lr_scheduler.StepLR(opt_func, step_size, gamma=weight_decay)

# train_loss, train_acc, val_acc, F_out, valtime, F_norm, Grad_norm, Coe_ten = train_val(epochs, max_lr, model, Beta, scheduler, device, train_loader, val_loader, 
#                   weight_decay, grad_clip, opt_func, b_size=batch_size)

model = CNN3(num_conv_layers = num_conv_layers,
    input_channels=1, out_channels_base=out_channels_base, 
    rate=rate, mode = mode,
    num_classes=10, batch_norm = batch_norm, pool = True).to(device)
    
opt_func = LARS(model.parameters(), lr=max_lr, momentum=0.9, weight_decay=1e-4)
#opt_func = torch.optim.SGD(model.parameters(), lr=max_lr, momentum=0.9,weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.StepLR(opt_func, step_size, gamma=weight_decay)

train_loss, train_acc, val_acc, F_out, valtime, F_norm, Grad_norm, GRAD, BETA = train_val(epochs, max_lr, model, Beta, scheduler, device, train_loader, val_loader,
                  weight_decay=weight_decay, grad_clip=grad_clip, opt_func=opt_func, b_size = batch_size)

from datetime import datetime
from pytz import timezone     

Ue = timezone('US/Eastern')
Ue_time = datetime.now(Ue)
time = Ue_time.strftime('%m-%d-%H-%M')
path = 'mnist-cnn-domi'+time
os.makedirs(path)
torch.save(GRAD, path+'/grad')
torch.save(BETA, path+'/beta')

plt.rcParams['text.usetex'] = True
plt.style.use('seaborn-paper')
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 150
t = [(i+1) for i in range(len(GRAD[:,0]))]
for i in range(5):
    plt.scatter(torch.log(GRAD[:,i]), torch.log(abs(BETA[:,i])), c=t, cmap='viridis_r')
    plt.colorbar()
    plt.locator_params(axis='x', nbins=8)
    axes = plt.gca()
    plt.xlabel('$\log(\|f(x)\|)$', color='k')
    plt.ylabel('$\log(\|\\nabla f(x)\|)$', color='k')
    axes = plt.gca()
    axes.xaxis.label.set_size(18)
    axes.yaxis.label.set_size(18)
    plt.xticks(color='k', fontsize=14)
    plt.yticks(color='k', fontsize=14)
    plt.tight_layout()
    plt.savefig(path+'/fig'+str(i)+'.pdf')
    plt.clf()

