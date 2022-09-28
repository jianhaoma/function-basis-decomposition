
# imports
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from flash.core.optimizers import LARS
from nngeometry.generator import Jacobian
from nngeometry.layercollection import LayerCollection
from torchvision.datasets import MNIST

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Download  dataset and load data
dataset = MNIST(root='data/', download=True)
vali_dataset = MNIST(root='data/', train=False)
train_ds = MNIST(root='data/', 
                train=True,
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
val_ds = MNIST(root='data/', 
                train=False,
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

# specify device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define convolutional network with varying depth 
def convnet(layers) -> nn.Module:
    if layers == 2:
        width = [1, 256, 64]
        modules = []
        for i in range(layers):
          modules.extend([
                      nn.Conv2d(width[i], width[i+1], kernel_size=3, stride=1, padding=1),
                      torch.nn.ReLU(),
                      torch.nn.MaxPool2d(2),])
        modules.append(nn.Flatten())
        modules.append(nn.Linear(3136, 10, bias=False))
        return nn.Sequential(*modules)

    elif layers == 3:
        width = [1, 256, 128, 64]
        modules = []
        for i in range(layers):
          modules.extend([
                      nn.Conv2d(width[i], width[i+1], kernel_size=3, stride=1, padding=1),
                      torch.nn.ReLU(),
                      torch.nn.MaxPool2d(2),])
        modules.append(nn.Flatten())
        modules.append(nn.Linear(576, 10, bias=False))
        return nn.Sequential(*modules)
    
    elif layers == 4:
        width = [1, 256, 128, 64, 32]
        modules = []
        for i in range(layers):
          modules.extend([
                      nn.Conv2d(width[i], width[i+1], kernel_size=3, stride=1, padding=1),
                      torch.nn.ReLU(),
                      torch.nn.MaxPool2d(2),])
        modules.append(nn.Flatten())
        modules.append(nn.Linear(32, 10, bias=False))
        return nn.Sequential(*modules)

# Define a hook
features = {}
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

# validation step for the model
def vali_step(model, device, val_loader, compute_phi=False):
    model.eval()
    FEATS = []
    num_batch = len(val_loader)
    f_out = torch.empty((0, 10))
    vacc = 0
    total = 0
    correct = 0
    for batch in val_loader:
      images, labels = batch
      images, labels = images.to(device), labels.to(device)
      output = model(images)
      total += labels.size(0)
      _, predicted = torch.max(output.data, 1)
      correct += (predicted == labels).sum().item()
      labels=F.one_hot(labels.to(torch.int64), 10).float()   
      f_out = torch.vstack((f_out, output.detach().cpu()))
      if compute_phi:
        FEATS.append(features['phi'].cpu().numpy())
    vacc = correct/total
    if compute_phi:
      phi = torch.flatten(torch.from_numpy(FEATS[0]), 1)
      for i in range(num_batch-1):
        phi = torch.vstack((phi, torch.flatten(torch.from_numpy(FEATS[i+1]), 1)))
    else:
      phi = None
    return vacc, phi, f_out

def train(epochs, model, scheduler, device, train_loader, val_loader, grad_clip=None, opt_func=torch.optim.SGD):
    F_out = {}
    train_acc = []
    val_acc = []
    train_loss = []
    valtime = []
    iter = 0

    model.eval()
    vacc, _, f_out = vali_step(model, device, val_loader, compute_phi=False)
    val_acc.append(vacc)
    F_out[0] = f_out
    valtime.append(0)
    for epoch in range(epochs):

        # optimizer
        optimizer = opt_func
        total = 0
        correct = 0
        for batch in train_loader:
          model.train()
          images, labels = batch 
          images, labels = images.to(device), labels.to(device)
          out = model(images)                  
          _, predicted = torch.max(out.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
          train_acc.append(correct/total)
          labels=F.one_hot(labels.to(torch.int64), 10).float()
          loss = F.mse_loss(out, labels)
          loss.backward()
          train_loss.append(loss.detach())
          # Gradient clipping
          if grad_clip: 
              nn.utils.clip_grad_value_(model.parameters(), grad_clip)
          optimizer.step()
          optimizer.zero_grad()

          # Validation phase
          if iter%20 == 0 and iter<2001:
                vacc, _, f_out = vali_step(model, device, val_loader, compute_phi=False)
                F_out[iter+1] = f_out
                val_acc.append(vacc)
                valtime.append(iter+1)
          iter += 1
        scheduler.step()
    # Register a hook for the last layer
    second_to_last = list(model.__dict__['_modules'].keys())[-2]
    handle = getattr(model, second_to_last).register_forward_hook(get_features('phi'))
    model.eval()
    with torch.no_grad():
        vacc, phi, _ = vali_step(model, device, val_loader, compute_phi=True)
        handle.remove()
        beta = list(model.state_dict().items())[-1][1].detach().cpu()
    return train_loss, train_acc, val_acc, phi, beta, F_out, valtime


# experiment setting: 
num_conv_layers = 3  #depth of network
batch_size = 256
test_size = batch_size
epochs = 20    
max_lr = 0.05  
grad_clip = 0.1
# weight_decay = 1e-4
step_size = 50
weight_decay = 0.33
train_loader = DataLoader(train_ds, batch_size, shuffle=True) 
val_loader = DataLoader(val_ds, test_size)

model = convnet(num_conv_layers).to(device)

opt_func = LARS(model.parameters(), lr=max_lr, momentum=0.9, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.StepLR(opt_func, step_size, gamma=weight_decay)

train_loss, train_acc, val_acc, phi, W, F_out, valtime = train(epochs, model, scheduler, device, train_loader, val_loader, grad_clip, opt_func)
train_loss = [x.detach().cpu().numpy() for x in train_loss]

from datetime import datetime
from pytz import timezone     

Ue = timezone('US/Eastern')
Ue_time = datetime.now(Ue)
time = Ue_time.strftime('%m-%d-%H-%M')

U, S, V = torch.svd(phi)

beta_star = torch.matmul(W, V)
Coe = {}
for i in range(5):
    Coe[i+1] = []
keys = list(F_out.keys())
Iter = len(F_out.keys())
A = torch.matmul(V, torch.diag(S))
beta_val = torch.matmul(W, A)/100

u, s, v = torch.svd(beta_val)
m, n = beta_val.shape

for iter in range(Iter):
    beta_val = torch.matmul(F_out[keys[iter]].t(), U)/100
    for i in range(5):
        coe = torch.mul(beta_val, u[:, i].reshape((m, 1)) @ v[:, i].reshape((n, 1)).T).sum()
        Coe[i+1].append(coe)

plt.rcParams['text.usetex'] = True
plt.style.use('seaborn-paper')
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 150
for i in range(5):
    i_1 = i + 1
    plt.plot(keys, Coe[i+1],
            linewidth=2,
            label=r'$\beta_{%s}$' % (i_1))
plt.xlim(0,2000)
plt.ylim(0,)
plt.locator_params(axis='x', nbins=8)
plt.legend(prop={'size': 16})
axes = plt.gca()
plt.xlabel("iteration", color='k')
plt.ylabel("scale", color='k')
plt.legend(loc='lower right', prop={'size': 22})
axes = plt.gca()
axes.xaxis.label.set_size(20)
axes.yaxis.label.set_size(20)
plt.xticks(color='k', fontsize=16)
plt.yticks(color='k', fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig('cnn3-mnist-coe.pdf')