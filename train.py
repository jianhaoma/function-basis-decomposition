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


def load_architecture(arch_id: str, num_chann: int, num_classes: int, pic_size: int) -> nn.Module:
    #  ======   fully-connected networks =======
    if arch_id == 'fc-relu':
        return fully_connected_net(num_chann, num_classes, pic_size, [200, 200], 'relu', bias=True)
    elif arch_id == 'fc-elu':
        return fully_connected_net(num_chann, num_classes, pic_size, [200, 200], 'elu', bias=True)
    elif arch_id == 'fc-tanh':
        return fully_connected_net(num_chann, num_classes, pic_size, [200, 200], 'tanh', bias=True)
    elif arch_id == 'fc-hardtanh':
        return fully_connected_net(num_chann, num_classes, pic_size, [200, 200], 'hardtanh', bias=True)
    elif arch_id == 'fc-softplus':
        return fully_connected_net(num_chann, num_classes, pic_size, [200, 200], 'softplus', bias=True)

    #  ======   convolutional networks =======
    elif arch_id == 'cnn-relu':
        return convnet(num_chann, num_classes, pic_size, [pic_size, pic_size], activation='relu', pooling='max', bias=True)
    elif arch_id == 'cnn-elu':
        return convnet(num_chann, num_classes, pic_size, [pic_size, pic_size], activation='elu', pooling='max', bias=True)
    elif arch_id == 'cnn-tanh':
        return convnet(num_chann, num_classes, pic_size, [pic_size, pic_size], activation='tanh', pooling='max', bias=True)
    elif arch_id == 'cnn-avgpool-relu':
        return convnet(num_chann, num_classes, pic_size, [pic_size, pic_size], activation='relu', pooling='average', bias=True)
    elif arch_id == 'cnn-avgpool-elu':
        return convnet(num_chann, num_classes, pic_size, [pic_size, pic_size], activation='elu', pooling='average', bias=True)
    elif arch_id == 'cnn-avgpool-tanh':
        return convnet(num_chann, num_classes, pic_size, [pic_size, pic_size], activation='tanh', pooling='average', bias=True)

    #  ======   convolutional networks with BN =======
    elif arch_id == 'cnn-bn-relu':
        return convnet_bn(num_chann, num_classes, pic_size, [pic_size, pic_size], activation='relu', pooling='max', bias=True)
    elif arch_id == 'cnn-bn-elu':
        return convnet_bn(num_chann, num_classes, pic_size, [pic_size, pic_size], activation='elu', pooling='max', bias=True)
    elif arch_id == 'cnn-bn-tanh':
        return convnet_bn(num_chann, num_classes, pic_size, [pic_size, pic_size], activation='tanh', pooling='max', bias=True)

    # ======= vary depth =======
    elif arch_id == 'fc-tanh-depth1':
        return fully_connected_net(num_chann, num_classes, pic_size, [200], 'tanh', bias=True)
    elif arch_id == 'fc-tanh-depth2':
        return fully_connected_net(num_chann, num_classes, pic_size, [200, 200], 'tanh', bias=True)
    elif arch_id == 'fc-tanh-depth3':
        return fully_connected_net(num_chann, num_classes, pic_size, [200, 200, 200], 'tanh', bias=True)
    elif arch_id == 'fc-tanh-depth4':
        return fully_connected_net(num_chann, num_classes, pic_size, [200, 200, 200, 200], 'tanh', bias=True)

    # ======= applicable NNs ======= 
    elif arch_id == 'resnet18':
        model = torchvision.models.resnet18(pretrained=False, num_classes=10)
        model.conv1 = nn.Conv2d(3,
                            64,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=(1, 1),
                            bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=False)
        return model
    elif arch_id == 'vgg11':
        model = torchvision.models.vgg11(pretrained=False, num_classes=10)
        model.conv1 = nn.Conv2d(3,
                                64,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding=(1, 1),
                                bias=False)
        model.maxpool = nn.Identity()
        # initial value
        return model
    elif arch_id == 'alexnet':
        model = torchvision.models.alexnet(pretrained=False, num_classes=10)
        model.features[0] = nn.Conv2d(3,
                            64,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=(1, 1),
                            bias=False)
        model.maxpool = nn.Identity()
        return model
    elif arch_id == 'vision-transformer':
        model = torchvision.models.vit_b_16(pretrained=False, num_classes=10)
        model.conv_proj = nn.Conv2d(3,
                                64,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding=(1, 1),
                                bias=False)
        model.maxpool = nn.Identity()
        model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes, bias=False)
        return model
    elif arch_id == 'wide-resnet':
        model = torchvision.models.wide_resnet50_2(pretrained=False, num_classes=10)
        model.conv1 = nn.Conv2d(3,
                                64,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding=(1, 1),
                                bias=False)
        model.head = torch.nn.Linear(in_features=768, out_features=num_classes, bias=False)
        return model
    else:
        raise NotImplementedError('unknown model: '+arch_id)

# load training data
def load_train_data(batch_size, dataset, num_workers):

    if dataset == 'Mnist':
        train_set = torchvision.datasets.MNIST(root='./data',
                                                train=True,
                                                download=True,
                                                transform=transforms.ToTensor())      
    
    elif dataset == 'Cifar10':
        param_mean = (0.4914, 0.4822, 0.4465)
        param_std = (0.2471, 0.2435, 0.2616
                 )
        transform_train = transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(param_mean, param_std),
        ])

        train_set = torchvision.datasets.CIFAR10(root='./data',
                                                train=True,
                                                download=True,
                                                transform=transform_train)

    elif dataset == 'Cifar100':
        param_mean = (0.5071, 0.4867, 0.4408)
        param_std = (0.2675, 0.2565, 0.2761)
                 
        transform_train = transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(param_mean, param_std),
        ])

        train_set = torchvision.datasets.CIFAR100(root='./data',
                                                train=True,
                                                download=True,
                                                transform=transform_train)
                                                
    elif dataset == 'Imagenet':
        param_mean = [0.485, 0.456, 0.406]
        param_std = [0.229, 0.224, 0.225]
                 
        transform_train = transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(param_mean, param_std),
        ])

        train_set = torchvision.datasets.ImageNet(root='./data',
                                                train=True,
                                                download=True,
                                                transform=transform_train)
    else:
        raise NotImplementedError('unknown dataset: '+dataset)                                                                                

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers)

    return train_loader

# load test data and validation data (here, test == validation)
def load_test_data(batch_size, dataset, num_workers):

    if dataset == 'Mnist':
        test_set = torchvision.datasets.MNIST(root='./data',
                                                train=False,
                                                download=True,
                                                transform=transforms.ToTensor()) 

    elif dataset == 'Cifar10':
        param_mean = (0.4914, 0.4822, 0.4465)
        param_std = (0.2471, 0.2435, 0.2616)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(param_mean, param_std),
        ])

    test_set = torchvision.datasets.CIFAR10(root='./data',
                                            train=False,
                                            download=True,
                                            transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=num_workers)                                              

    return test_loader

# Define a hook
features = {}
def get_features(name):

    def hook(model, input, output):
        features[name] = output.detach()

    return hook

def initial_test(model, val_loader, device, num_classes):
    # validation phase
    model.eval()
    f_out = torch.empty((0, num_classes))
    with torch.no_grad():
        correct, total = 0, 0
        num_batch = len(val_loader)
        for _, data in enumerate(val_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            f_out = torch.vstack((f_out, outputs.detach().cpu()))
        weight = list(model.state_dict().items())[-1][1].detach().cpu()
    return f_out, weight, correct/total

def init_scale(model, scale):
    state_dict = model.state_dict()
    for i in range(len(state_dict.keys())):
        name = list(model.state_dict())[i]
        state_dict[name] = state_dict[name]*scale
    model.load_state_dict(state_dict)
    return model

def info(data):
    if data == 'Mnist':
        return 1, 10, 28
    elif data == 'Cifar100':
        return 3, 100, 32
    elif data == 'Imagenet':
        return 3, 1000, 256   
    elif data == 'Cifar10':
        return 3, 10, 32
    else:
        raise NotImplementedError('unknown dataset: '+data)

def model_train(model, device, train_loader, val_loader, EPOCH, criterion, args):
    model_name=args.model
    call_wandb=args.call_wandb
    if call_wandb:
        def train_log(loss, example_ct, epoch):
            # Where the magic happens
            wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
            print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")
        wandb.watch(model, criterion, log="all", log_freq=100)
    example_ct = 0  # number of examples seen
    batch_ct = 0
    best_acc = 0
    name = model_name
    F_out = torch.zeros((10000, 10, 1))
    train_loss = []

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.init_lr,
                                momentum=0.9,
                                weight_decay=1e-4)
    if args.warmup == 'exp':
        exp_rate = 1.05
        lambda1 = lambda epoch: args.init_lr*exp_rate**epoch
        end = int(np.log(args.lr/args.init_lr)/np.log(exp_rate))
        scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    else:
        lambda1 = lambda epoch: LR
        end = 0
        scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=args.decay_stepsize,
                                                gamma=args.decay_rate)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[end])

    for epoch in range(EPOCH):
        model.train()
        correct, sum_loss, total = 0, 0, 0
        for _, data in enumerate(train_loader, 0):
            # prepare data
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(F.softmax(outputs, 1),
                             F.one_hot(labels, num_classes=10).float())
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if call_wandb:
                example_ct += len(inputs)
                batch_ct += 1

                if ((batch_ct + 1) % 25) == 0:
                    train_log(loss, example_ct, epoch)
        train_loss.append(sum_loss)
        training_acc = correct / total
        train_acc.append(training_acc)
        if call_wandb:
            wandb.log({"train_accuracy": training_acc})

        # validation phase
        weight = list(model.state_dict().items())[-1][1].detach().cpu()
        weights = torch.cat([weights, torch.unsqueeze(weight, 2)], dim=2)
        if epoch == EPOCH - 1:
            feature_name = 'phi'
            if model_name == 'vgg11' or model_name == 'alexnet':
                handle = model.classifier[5].register_forward_hook(get_features(feature_name))
            elif model_name == 'vit':
                handle = model[2][4].register_forward_hook(get_features(feature_name))
            else:
                second_to_last = list(model.__dict__['_modules'].keys())[-2]
                handle = getattr(model, second_to_last).register_forward_hook(get_features(feature_name))
            FEATS = []
        
        model.eval()
        f_out = torch.empty((0, num_classes))
        if args.gradient==True:
            correct, total = 0, 0
            num_batch = len(val_loader)
            lc = LayerCollection.from_model(model)
            Jac = Jacobian(model, n_output=num_classes, centering=True, layer_collection=lc)
            grad_norm_ = 0
            for _, data in enumerate(val_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                f_out = torch.vstack((f_out, outputs.detach().cpu()))
                if epoch == EPOCH - 1:
                    FEATS.append(features[feature_name].cpu().numpy())
                grad_norm_ += torch.norm((Jac.get_jacobian([inputs, labels])).detach().cpu(), 'fro')**2
            F_out = torch.cat([F_out, torch.unsqueeze(f_out, 2)], dim=2)
            test_acc = correct / total
            test_acc_.append(test_acc)
            grad_norm.append(torch.sqrt(grad_norm_).detach().cpu()*1e-4)
            # test_accs.append(test_acc)
        else:
            with torch.no_grad():
                correct, total = 0, 0
                num_batch = len(val_loader)
                for _, data in enumerate(val_loader, 0):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    f_out = torch.vstack((f_out, outputs.detach().cpu()))
                    if epoch == EPOCH - 1:
                        FEATS.append(features[feature_name].cpu().numpy())
                F_out = torch.cat([F_out, torch.unsqueeze(f_out, 2)], dim=2)
                test_acc = correct / total
                test_acc_.append(test_acc)
        if call_wandb:
            wandb.log({"test_accuracy": test_acc})

        if epoch == EPOCH - 1:
            phi = torch.flatten(torch.from_numpy(FEATS[0]), 1)
            for i in range(num_batch - 1):
                phi = torch.vstack(
                    (phi, torch.flatten(torch.from_numpy(FEATS[i + 1]), 1)))
            handle.remove()
            beta = list(model.state_dict().items())[-1][1].detach().cpu()
        scheduler.step()

        # # plot
        # plt.plot(training_accs, linewidth=2, label='train acc')
        # plt.plot(test_accs, linewidth=2, label='test acc')
        # plt.legend(loc='best')
        # plt.xlim(0, epoch + 1)
        # plt.ylim(0, 1)
        # plt.xlabel("epoch")
        # plt.ylabel("accuracy")
        # plt.legend(prop={'size': 18})
        # axes = plt.gca()
        # axes.xaxis.label.set_size(18)
        # axes.yaxis.label.set_size(18)
        # plt.xticks(color='k', fontsize=14)
        # plt.yticks(color='k', fontsize=14)
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig('resnet_18_v2')
        # plt.clf()

    return model, best_acc, beta, phi, F_out, train_loss

def arg_parser():
    # parsers
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training VGG')
    parser.add_argument('--lr', default=1e-1, type=float, help='learning rate')
    parser.add_argument('--init_lr', default=1e-2, type=float, help='initial learning rate')
    parser.add_argument('--warmup', default='none', help='different warm up lr schemes')
    parser.add_argument('--warmtime', default=100, help='warm up lr time')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--batch_size', default=512, type=int, help='batch size')
    parser.add_argument('--num_epoch', default=310, type=int, help='number of epoch')
    parser.add_argument('--model', default='resnet18', help='type of model')
    parser.add_argument('--data', default='Cifar10', help='type of dataset')
    parser.add_argument('--decay_rate', default=0.33, type=float, help='the decay rate in each stepsize decay')
    parser.add_argument('--decay_stepsize', default=50, type=int, help='the decay time')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--num_run', default=0, type=int, help='the number of run')
    parser.add_argument('--call_wandb', default=False, type=bool, help='connect with wandb or not')
    parser.add_argument('--init_scale', default=1, type=float, help='scale model initial parameter')
    parser.add_argument('--loss_fn', default='mse_loss', help='loss type')
    args = parser.parse_args()
    return args

def main():
    
    args = arg_parser()
    # fix random seed
    clear_output()
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    from datetime import datetime
    from pytz import timezone     

    Ue = timezone('US/Eastern')
    Ue_time = datetime.now(Ue)
    time = Ue_time.strftime('%m-%d-%H-%M')
    if args.call_wandb:
        wandb.init(project=args.model+args.data, name=dir_name, 
           entity="incremental-learning-basis-decomposition")
    # Define hyperparameters
    EPOCH = args.num_epoch
    batch_size = args.batch_size
    num_run = args.num_run
    num_workers = 0
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck')
    LR = args.lr
    # load data and model to device
    param_mean = (0.4914, 0.4822, 0.4465)
    param_std = (0.2471, 0.2435, 0.2616
                 )  # param_mean, param_std = get_mean_and_std(dataset_path)
    train_loader = load_train_data(batch_size, param_mean, param_std,
                                   num_workers)
    val_loader = load_test_data(batch_size, param_mean, param_std, num_workers)
    device = get_default_device()
    model = load_architecture(args.model, args.data)
    # model = create_model().to(device)
    init_scale(model, args.init_scale).to(device)
    #reinitialize model parameters
    if args.loss_fn == 'mse_loss':
        criterion = nn.MSELoss()
    elif args.loss_fn == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    # Start Training model, train_dl, vali_dl, EPOCH, criterion, optimizer, scheduler

    init_f_out = initial_test(model, val_loader, device)
    model, best_acc, beta, Phi, F_out, train_loss = model_train(model, device, train_loader, val_loader, EPOCH, criterion, args)

    # save model
    if args.path != 'none':
        dir_name = args.path
    os.makedirs(dir_name)

    # plotting style
    plt.style.use('seaborn-paper')
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 150

    plt.plot(train_loss, linewidth=2, label='train loss')
    plt.locator_params(axis='x', nbins=8)
    plt.legend(prop={'size': 10})
    axes = plt.gca()
    plt.xlabel("epoch", color='k')
    plt.legend(loc='best', prop={'size': 18})
    plt.title('train loss')
    axes = plt.gca()
    axes.xaxis.label.set_size(18)
    axes.yaxis.label.set_size(18)
    plt.xticks(color='k', fontsize=14)
    plt.yticks(color='k', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(dir_name+'/'+args.model+'train_loss')
    plt.clf()

    U, S, V = torch.svd(Phi)
    beta_star = torch.matmul(beta, V)
    Coe = {}
    _, _, Iter = F_out.shape
    for i in range(20):
        u = torch.outer(beta_star[:, i], U[:, i])
        norm_u = torch.trace(u.t().mm(u)) / 10000
        u = u / norm_u
        coe = []
        for iter in range(Iter):
            coe.append((torch.trace(F_out[:, :, iter].mm(u))) / 10000)
        Coe[i + 1] = coe

    for i in range(5):
        i_1 = i + 1
        plt.plot(range(Iter), [x for x in Coe[i + 1]][0:Iter],
                linewidth=2,
                label=r'$\beta_{%s}$' % i_1)
    plt.locator_params(axis='x', nbins=8)
    plt.legend(prop={'size': 10})
    axes = plt.gca()
    plt.xlabel("epoch", color='k')
    plt.legend(loc='lower right', prop={'size': 18})
    plt.title('the {} run'.format(i+1))
    axes = plt.gca()
    axes.xaxis.label.set_size(18)
    axes.yaxis.label.set_size(18)
    plt.xticks(color='k', fontsize=14)
    plt.yticks(color='k', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(dir_name+'/'+args.model+'beta-5')
    plt.clf()

    plt.plot(test_acc_, linewidth=2, label='test acc')
    plt.locator_params(axis='x', nbins=8)
    plt.legend(prop={'size': 10})
    axes = plt.gca()
    plt.xlabel("epoch", color='k')
    plt.legend(loc='lower right', prop={'size': 18})
    plt.title('the {} run'.format(i+1))
    axes = plt.gca()
    axes.xaxis.label.set_size(18)
    axes.yaxis.label.set_size(18)
    plt.xticks(color='k', fontsize=14)
    plt.yticks(color='k', fontsize=14)
    plt.grid(True)
    plt.savefig(dir_name+'/'+args.model+'beta-20')
    return

if __name__ == "__main__":  
    main()
    #wide resnet consumes large memory. recommend small batch size(batch size=512 blows up 16g gpu). requires longer running time.
