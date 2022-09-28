# imports
import wandb
import flash
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from flash.core.optimizers import LARS
from flash.core.optimizers import LAMB
import os
import argparse
from nngeometry.generator import Jacobian
from nngeometry.layercollection import LayerCollection

def load_architecture(arch_id: str, num_chann: int, num_classes: int, pic_size: int) -> nn.Module:

    if arch_id == 'resnet18':
        model = torchvision.models.resnet18(num_classes=num_classes)
        model.conv1 = nn.Conv2d(num_chann,
                            2*pic_size,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=(1, 1),
                            bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=False)
        return model
    elif arch_id == 'resnet34':
        model = torchvision.models.resnet34(num_classes=num_classes)
        model.conv1 = nn.Conv2d(num_chann,
                            2*pic_size,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=(1, 1),
                            bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=False)
        return model
    elif arch_id == 'resnet50':
        model = torchvision.models.resnet50(num_classes=num_classes)
        model.conv1 = nn.Conv2d(num_chann,
                            2*pic_size,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=(1, 1),
                            bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=False)
        return model
    elif arch_id == 'vgg11':
        model = torchvision.models.vgg11(num_classes=num_classes)
        model.features[0] = nn.Conv2d(num_chann,
                                2*pic_size,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding=(1, 1),
                                bias=False)
        model.features[2] = nn.Identity()
        model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes, bias=False)
        # initial value
        return model
    elif arch_id == 'alexnet':
        model = torchvision.models.alexnet(num_classes=num_classes)
        model.features[0] = nn.Conv2d(num_chann,
                            2*pic_size,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=(1, 1),
                            bias=False)
        model.features[2] = nn.Identity()
        model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes, bias=False)
        return model
    elif arch_id == 'vit':
        class Residual(nn.Module):
            def __init__(self, *layers):
                super().__init__()
                self.residual = nn.Sequential(*layers)
                self.gamma = nn.Parameter(torch.zeros(1))
            
            def forward(self, x):
                return x + self.gamma * self.residual(x)

        class LayerNormChannels(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.norm = nn.LayerNorm(channels)
            
            def forward(self, x):
                x = x.transpose(1, -1)
                x = self.norm(x)
                x = x.transpose(-1, 1)
                return x

        class SelfAttention2d(nn.Module):
            def __init__(self, in_channels, out_channels, head_channels, shape):
                super().__init__()
                self.heads = out_channels // head_channels
                self.head_channels = head_channels
                self.scale = head_channels**-0.5
                
                self.to_keys = nn.Conv2d(in_channels, out_channels, 1)
                self.to_queries = nn.Conv2d(in_channels, out_channels, 1)
                self.to_values = nn.Conv2d(in_channels, out_channels, 1)
                self.unifyheads = nn.Conv2d(out_channels, out_channels, 1)
                
                height, width = shape
                self.pos_enc = nn.Parameter(torch.Tensor(self.heads, (2 * height - 1) * (2 * width - 1)))
                self.register_buffer("relative_indices", self.get_indices(height, width))
            
            def forward(self, x):
                b, _, h, w = x.shape
                
                keys = self.to_keys(x).view(b, self.heads, self.head_channels, -1)
                values = self.to_values(x).view(b, self.heads, self.head_channels, -1)
                queries = self.to_queries(x).view(b, self.heads, self.head_channels, -1)
                
                att = keys.transpose(-2, -1) @ queries
                
                indices = self.relative_indices.expand(self.heads, -1)
                rel_pos_enc = self.pos_enc.gather(-1, indices)
                rel_pos_enc = rel_pos_enc.unflatten(-1, (h * w, h * w))
                
                att = att * self.scale + rel_pos_enc
                att = F.softmax(att, dim=-2)
                
                out = values @ att
                out = out.view(b, -1, h, w)
                out = self.unifyheads(out)
                return out
            
            @staticmethod
            def get_indices(h, w):
                y = torch.arange(h, dtype=torch.long)
                x = torch.arange(w, dtype=torch.long)
                
                y1, x1, y2, x2 = torch.meshgrid(y, x, y, x, indexing='ij')
                indices = (y1 - y2 + h - 1) * (2 * w - 1) + x1 - x2 + w - 1
                indices = indices.flatten()
                
                return indices

        class FeedForward(nn.Sequential):
            def __init__(self, in_channels, out_channels, mult=4):
                hidden_channels = in_channels * mult
                super().__init__(
                    nn.Conv2d(in_channels, hidden_channels, 1),
                    nn.GELU(),
                    nn.Conv2d(hidden_channels, out_channels, 1)   
                )

        class TransformerBlock(nn.Sequential):
            def __init__(self, channels, head_channels, shape, p_drop=0.):
                super().__init__(
                    Residual(
                        LayerNormChannels(channels),
                        SelfAttention2d(channels, channels, head_channels, shape),
                        nn.Dropout(p_drop)
                    ),
                    Residual(
                        LayerNormChannels(channels),
                        FeedForward(channels, channels),
                        nn.Dropout(p_drop)
                    )
                )

        class TransformerStack(nn.Sequential):
            def __init__(self, num_blocks, channels, head_channels, shape, p_drop=0.):
                layers = [TransformerBlock(channels, head_channels, shape, p_drop) for _ in range(num_blocks)]
                super().__init__(*layers)

        """Embedding of patches"""

        class ToPatches(nn.Sequential):
            def __init__(self, in_channels, channels, patch_size, hidden_channels=32):
                super().__init__(
                    nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(hidden_channels, channels, patch_size, stride=patch_size)
                )

        class AddPositionEmbedding(nn.Module):
            def __init__(self, channels, shape):
                super().__init__()
                self.pos_embedding = nn.Parameter(torch.Tensor(channels, *shape))
            
            def forward(self, x):
                return x + self.pos_embedding

        class ToEmbedding(nn.Sequential):
            def __init__(self, in_channels, channels, patch_size, shape, p_drop=0.):
                super().__init__(
                    ToPatches(in_channels, channels, patch_size),
                    AddPositionEmbedding(channels, shape),
                    nn.Dropout(p_drop)
                )

        """Main model"""

        class Head(nn.Sequential):
            def __init__(self, in_channels, classes, p_drop=0.):
                super().__init__(
                    LayerNormChannels(in_channels),
                    nn.GELU(),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Dropout(p_drop),
                    nn.Linear(in_channels, classes)
                )

        class RelViT(nn.Sequential):
            def __init__(self, classes, image_size, channels, head_channels, num_blocks, patch_size,
                        in_channels=3, emb_p_drop=0., trans_p_drop=0., head_p_drop=0.):
                reduced_size = image_size // patch_size
                shape = (reduced_size, reduced_size)
                super().__init__(
                    ToEmbedding(in_channels, channels, patch_size, shape, emb_p_drop),
                    TransformerStack(num_blocks, channels, head_channels, shape, trans_p_drop),
                    Head(channels, classes, head_p_drop)
                )
                self.reset_parameters()
            
            def reset_parameters(self):
                for m in self.modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        nn.init.kaiming_normal_(m.weight)
                        if m.bias is not None: nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.LayerNorm):
                        nn.init.constant_(m.weight, 1.)
                        nn.init.zeros_(m.bias)
                    elif isinstance(m, AddPositionEmbedding):
                        nn.init.normal_(m.pos_embedding, mean=0.0, std=0.02)
                    elif isinstance(m, SelfAttention2d):
                        nn.init.normal_(m.pos_enc, mean=0.0, std=0.02)
                    elif isinstance(m, Residual):
                        nn.init.zeros_(m.gamma)
            
            def separate_parameters(self):
                parameters_decay = set()
                parameters_no_decay = set()
                modules_weight_decay = (nn.Linear, nn.Conv2d)
                modules_no_weight_decay = (nn.LayerNorm,)

                for m_name, m in self.named_modules():
                    for param_name, param in m.named_parameters():
                        full_param_name = f"{m_name}.{param_name}" if m_name else param_name

                        if isinstance(m, modules_no_weight_decay):
                            parameters_no_decay.add(full_param_name)
                        elif param_name.endswith("bias"):
                            parameters_no_decay.add(full_param_name)
                        elif isinstance(m, Residual) and param_name.endswith("gamma"):
                            parameters_no_decay.add(full_param_name)
                        elif isinstance(m, AddPositionEmbedding) and param_name.endswith("pos_embedding"):
                            parameters_no_decay.add(full_param_name)
                        elif isinstance(m, SelfAttention2d) and param_name.endswith("pos_enc"):
                            parameters_no_decay.add(full_param_name)
                        elif isinstance(m, modules_weight_decay):
                            parameters_decay.add(full_param_name)

                # sanity check
                assert len(parameters_decay & parameters_no_decay) == 0
                assert len(parameters_decay) + len(parameters_no_decay) == len(list(model.parameters()))

                return parameters_decay, parameters_no_decay

        model = RelViT(num_classes, 32, channels=256, head_channels=32, num_blocks=8, patch_size=2,emb_p_drop=0., trans_p_drop=0., head_p_drop=0.3)
        model[2][5] = torch.nn.Linear(in_features=256, out_features=num_classes, bias=False)
        return model
    elif arch_id == 'wide-resnet':
        model = torchvision.models.wide_resnet50_2(num_classes=num_classes)
        model.conv1 = nn.Conv2d(num_chann,
                                2*pic_size,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding=(1, 1),
                                bias=False)
        model.maxpool = nn.Identity()
        model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes, bias=False)
        return model
    elif arch_id == 'swim-net':
        model = torchvision.models.swin_t(num_classes=num_classes)
        model.features[0][0] = nn.Conv2d(num_chann,
                                2*pic_size,
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

    if dataset == 'Cifar10':
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
                                                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

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

    if dataset == 'Cifar10':
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
    elif dataset == 'Cifar100':
        param_mean = (0.5071, 0.4867, 0.4408)
        param_std = (0.2675, 0.2565, 0.2761)
                 
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(param_mean, param_std),
        ])

        test_set = torchvision.datasets.CIFAR100(root='./data',
                                                train=False,
                                                download=True,
                                                transform=transform_test)                                    
    
    elif dataset == 'Imagenet':
        param_mean = [0.485, 0.456, 0.406]
        param_std = [0.229, 0.224, 0.225]
                 
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(param_mean, param_std),
        ])

        train_set = torchvision.datasets.ImageNet(root='./data',
                                                train=True,
                                                download=True,
                                                transform=transform_train)  
    else:
        raise NotImplementedError('unknown dataset: '+dataset)                                            
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
    if data == 'Cifar100':
        return 3, 100, 32
    elif data == 'Imagenet':
        return 3, 1000, 256   
    elif data == 'Cifar10':
        return 3, 10, 32
    else:
        raise NotImplementedError('unknown dataset: '+data)

def get_optimizer(model, learning_rate, weight_decay):
    param_dict = {pn: p for pn, p in model.named_parameters()}
    parameters_decay, parameters_no_decay = model.separate_parameters()
    
    optim_groups = [
        {"params": [param_dict[pn] for pn in parameters_decay], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in parameters_no_decay], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
    return optimizer

def model_train(args):
    # epoch; batch size
    EPOCH = args.num_epoch
    batch_size = args.batch_size
    
    # load data and model to device
    train_loader = load_train_data(batch_size, args.data,
                                   num_workers=0)
    if args.gradient:
        val_loader = load_test_data(24, args.data, num_workers=0)
    else:
        val_loader = load_test_data(batch_size, args.data, num_workers=0)
    
    num_chann, num_classes, pic_size = info(args.data)
    model_name = args.model
    model = load_architecture(model_name, num_chann, num_classes, pic_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # initialization scale
    init_scale(model, args.init_scale).to(device)

    # loss func
    if args.loss_fn == 'mse_loss':
        criterion = nn.MSELoss()
    elif args.loss_fn == 'ce_loss':
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError('unknown loss func: '+args.loss_fn)
    LR = args.lr_setting[0] 

    # optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                                weight_decay=1e-4)
    elif args.optimizer == 'lars':
        optimizer = LARS(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    elif args.optimizer == 'lamb':
        optimizer = LAMB(model.parameters(), lr=LR)
    elif args.optimizer == 'adamw':
        if args.model == 'vit' and args.data == 'Cifar10':
            optimizer = get_optimizer(model, LR, weight_decay=0.1)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    else:
        raise NotImplementedError('unknown optimizer: '+args.optimizer)
    # learining rate; warm up; weight decay
    if args.lr_setting[1]>0:
        init_lr = args.lr_setting[2]
        warmtime = args.lr_setting[1]
        lin_rate = (LR - init_lr)/(warmtime-1)
        lambda1 = lambda epoch: init_lr+lin_rate*epoch
        end = warmtime
        scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    else:
        lambda1 = lambda epoch: LR
        end = 0
        scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=args.decay_stepsize,
                                                gamma=args.decay_rate)
    if args.scheduler == 'default':
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[end])
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR,
                                             steps_per_epoch=len(train_loader), epochs=EPOCH)
    #
    example_ct = 0  # number of examples seen
    batch_ct = 0
    F_out = torch.zeros((10000, num_classes, 1))
    train_loss = []
    train_acc = []
    test_acc_ = []
    grad_norm = []
    call_wandb=args.call_wandb
    k, M = args.k_M
    if call_wandb:
        def train_log(loss, example_ct, epoch):
            # Where the magic happens
            wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
            print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")
        wandb.watch(model, criterion, log="all", log_freq=100)

    # Start Training model, train_dl, vali_dl, EPOCH, criterion, optimizer, scheduler
    init_f_out, weight, acc = initial_test(model, val_loader, device, num_classes)
    weights = torch.unsqueeze(weight, dim=2)
    test_acc_.append(acc)
    for epoch in range(EPOCH):
        model.train()
        correct, sum_loss, total = 0, 0, 0
        for _, data in enumerate(train_loader, 0):
            # prepare data
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if args.data == 'Cifar100' and args.loss_fn == 'mse_loss':
                one_hot = F.one_hot(labels, num_classes=num_classes).float()
                ones = torch.ones_like(one_hot)
                weight = k * one_hot + 1 * ones
                loss = (weight * (outputs - M * one_hot)**2).mean()
            else:
                loss = criterion(outputs,
                             F.one_hot(labels, num_classes=num_classes).float())
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
    F_out[:, :, 0] = init_f_out
    return model.state_dict(), beta, phi, F_out, train_loss, train_acc, test_acc_, grad_norm, weights

def arg_parser():
    # parsers
    parser = argparse.ArgumentParser(description='PyTorch Incremental Learning Experiments')
    parser.add_argument('--model', default='resnet18', help='type of model')
    parser.add_argument('--init_scale', default=1, type=float, help='scale model initial parameter')
    parser.add_argument('--data', default='Cifar10', help='type of dataset')
    parser.add_argument('--loss_fn', default='mse_loss', help='loss type')
    parser.add_argument('--optimizer', default='sgd', help='optimizer type')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--num_epoch', default=310, type=int, help='number of epoch')
    parser.add_argument('--lr_setting', default=[1e-1, 5, 1e-2], nargs='*', type = float, help='settings for tuning learning rate: [max_learning rate, warm_up epochsinitial_lr]')
    parser.add_argument('--decay_rate', default=0.33, type=float, help='the decay rate in each stepsize decay')
    parser.add_argument('--decay_stepsize', default=50, type=int, help='the decay time')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--num_run', default=0, type=int, help='the number of run')
    parser.add_argument('--call_wandb', default=False, action='store_true', help='connect with wandb or not')
    parser.add_argument('--no-call_wandb', dest='call_wandb', action='store_false')
    parser.add_argument('--gradient', default=False, action='store_true', help='document changes with gradient')
    parser.add_argument('--no-gradient', dest='gradient', action='store_false')
    parser.add_argument('--path', default='none', type=str, help='saving path')
    parser.add_argument('--log', default='essential', type=str, help='how much we want to document')
    parser.add_argument('--k_M', default=[5, 5], nargs='*', type = float, help='loss augumenting for cifar-100 data: weight = k * one_hot + 1 * ones, loss = (weight * (outputs - M * one_hot)**2).mean()')
    parser.add_argument('--scheduler', default='default', type=str, help='what scheduler for vit')

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

    dir_name = 'nb'+args.model+'-init-scale'+str(args.init_scale)+'-data'+args.data+'-ep'+str(args.num_epoch)+'-bs'+str(args.batch_size)+'-lr'+str(args.lr_setting[0])+'wp_epoch'+str(args.lr_setting[1])+'-init_lr_wp'+str(args.lr_setting[2])+'-'+args.loss_fn+'-weight_dec'+str(args.decay_rate)+'per'+str(args.decay_stepsize)+'-opt'+args.optimizer+'k'+str(args.k_M[0])+'M'+str(args.k_M[1])+'-schedu'+args.scheduler+time
    if args.call_wandb:
        wandb.init(project=args.model+args.data, name=dir_name, 
           entity="incremental-learning-basis-decomposition")
        wandb.config.update(args)
    
    _, beta, Phi, F_out, train_loss, train_acc, test_acc_, grad_norm, weights= model_train(args)

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
    plt.savefig(dir_name+'/'+args.model+'train_loss.pdf')
    plt.clf()

    plt.plot(train_acc, linewidth=2, label='train acc')
    plt.locator_params(axis='x', nbins=8)
    plt.legend(prop={'size': 10})
    axes = plt.gca()
    plt.xlabel("epoch", color='k')
    plt.legend(loc='best', prop={'size': 18})
    plt.title('training accuracy')
    axes = plt.gca()
    axes.xaxis.label.set_size(18)
    axes.yaxis.label.set_size(18)
    plt.xticks(color='k', fontsize=14)
    plt.yticks(color='k', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(dir_name+'/'+args.model+'train_acc')
    plt.savefig(dir_name+'/'+args.model+'train_acc.pdf')
    plt.clf()

    plt.plot(test_acc_, linewidth=2, label='test acc')
    plt.locator_params(axis='x', nbins=8)
    plt.legend(prop={'size': 10})
    axes = plt.gca()
    plt.xlabel("epoch", color='k')
    plt.legend(loc='best', prop={'size': 18})
    plt.title('test acc')
    axes = plt.gca()
    axes.xaxis.label.set_size(18)
    axes.yaxis.label.set_size(18)
    plt.xticks(color='k', fontsize=14)
    plt.yticks(color='k', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(dir_name+'/'+args.model+'test_acc')
    plt.savefig(dir_name+'/'+args.model+'test_acc.pdf')
    plt.clf()

    if args.gradient:
        keys = list(F_out.keys())
        fn_norm = []
        for i in range(len(keys)):
            fn_norm.append(1e-4*torch.norm(F_out[keys[i]], 'fro'))

        plt.plot([torch.log(x) for x in fn_norm], [torch.log(x) for x in grad_norm], linewidth=2)
        plt.locator_params(axis='x', nbins=8)
        plt.legend(prop={'size': 10})
        axes = plt.gca()
        plt.xlabel("log(output_norm)", color='k')
        plt.ylabel("log(output_grad_norm)", color='k')
        plt.legend(loc='best', prop={'size': 18})
        plt.title('grad_fn')
        axes = plt.gca()
        axes.xaxis.label.set_size(18)
        axes.yaxis.label.set_size(18)
        plt.xticks(color='k', fontsize=14)
        plt.yticks(color='k', fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        if args.call_wandb:
            wandb.log({"beta_5": plt.gcf()})
        plt.savefig(dir_name+'/'+args.model+'grad_fn')
        plt.savefig(dir_name+'/'+args.model+'grad_fn.pdf')
        plt.clf()

    U, S, V = torch.svd(Phi)
    beta_star = torch.matmul(beta, V)
    Coe = {}
    for i in range(5):
        Coe[i+1] = []
    _, _, Iter = weights.shape
    A = torch.matmul(V, torch.diag(S))
    beta_val = torch.matmul(beta, A)/100

    u, s, v = torch.svd(beta_val)
    m, n = beta_val.shape

    # spars = list(torch.norm(beta_val, dim=0))
    # top5 = sorted(range(len(spars)), key=lambda i: spars[i], reverse=True)[:5]
    
    for iter in range(Iter):
    
        beta_val = torch.matmul(F_out[:, :, iter].t(), U)/100
        # beta_norm = torch.norm(beta_val, dim=0)
        for i in range(5):
            coe = torch.mul(beta_val, u[:, i].reshape((m, 1)) @ v[:, i].reshape((n, 1)).T).sum()
            Coe[i+1].append(coe)
            if args.call_wandb:
                wandb.log({'epoch':iter+1, 'beta_'+str(i+1): coe})

    data_dict = {'test_acc':test_acc_, 'train_acc': train_acc, 'train_loss':train_loss, 'grad_norm': grad_norm, 'svd_s': S, 'spars': s}
    np.save(dir_name+'/data_dict.npy', data_dict)
    np.save(dir_name+'/coe.npy', Coe) 
    if args.log == 'detail':
        torch.save(Phi, dir_name+'/Phi')
        torch.save(F_out, dir_name+'/F_out')
        torch.save(beta, dir_name+'/beta')

    for i in range(5):
        i_1 = i + 1
        plt.plot(range(Iter), [x for x in Coe[i + 1]][0:Iter],
                linewidth=2,
                label=r'$\beta_{%s}$' % i_1)
    plt.xlim(0,300)
    plt.ylim(0,)
    plt.locator_params(axis='x', nbins=8)
    plt.legend(prop={'size': 16})
    axes = plt.gca()
    plt.xlabel("epoch", color='k')
    plt.ylabel("scale", color='k')
    plt.legend(loc='lower right', prop={'size': 22})
    axes = plt.gca()
    axes.xaxis.label.set_size(20)
    axes.yaxis.label.set_size(20)
    plt.xticks(color='k', fontsize=16)
    plt.yticks(color='k', fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    if args.call_wandb:
        wandb.log({"beta_5": plt.gcf()})
    plt.savefig(dir_name+'/'+args.model+'beta-5')
    plt.savefig(dir_name+'/'+args.model+'beta-5.pdf')
    plt.clf()

    return

if __name__ == "__main__":  
    main()
   
