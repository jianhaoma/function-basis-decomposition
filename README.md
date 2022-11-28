# Basis function decomposition for neural networks

This repository contains all codes for the paper:

[Behind the Scenes of Gradient Descent: A Trajectory Analysis via Basis Function Decomposition.](https://arxiv.org/abs/2210.00346)

## Incremental learning phenomenon


## Code
- `grad_indep.py`
`grad_indep.py` gives Figure 2($b$) in the paper. It captures the maximum absolute value of the inner product between $\nabla\beta_{i}(\theta_{t})$ and $\nabla\beta_{j}(\theta_{t})$, per 10 iterations for the first 100 iterations. We use a 3-block CNN and MNIST dataset. Use `python -run grad_indep.py` to see the output.

- `beta_domin.py`
`beta_domin.py` gives figure 2($c$) in our paper. It captures the log relationship between $|\beta_{i}(\theta_{t})|$ and $|\nabla\beta_{i}\theta_{t}|$ for the first 400 iterations. We use a 3-block CNN and MNIST dataset. Use `python -run beta_domin.py` to see the output.

- `small-init.py`
`small-init.py` gives Figure 2($a$) in our paper. It calculates the initial $|\beta_{i}(\theta_{0})|$ and is indexed by the coefficients of the final A-CK. Use `python -run small-init.py` to see the output.

- `mnist.py`
`mnist.py` gives Figure 3($a$), ($b$) and ($c$) in our paper. It traces the trajectory of CNN with different depths on the MNIST dataset. Set `num_conv_layers` equal to 2,3,4 for results of CNN with different depths. For the 4-block CNN, please change `seed=64` to get the same figure as in our paper.

## `train.py`
`train.py` gives Figure 1($d$)($e$)($f$), Figure 3($d$)($e$)($f$), Figure 5,6,7, and 8. It traces the optimization trajectories of various networks (ResNet, ViT, AlexNet, VGG, etc.) on the CIFAR-10 and CIFAR-100 datasets. It also returns information for the training process, such as training/testing loss, training/testing accuracy etc.

How to run `train.py`. 
Here is a running example for our code.
`srun -l python3 train.py --no-call_wandb --model='alexnet' --init_scale=1 --data='Cifar10' --loss_fn='mse_loss' --optimizer='lars' --batch_size=512 --num_epoch=300 --lr_setting 2 0 1e-2 --decay_rate=0.33 --decay_stepsize=50 --seed=0 --num_run=0 --no-gradient`

- `--model`: model type of the algorithm. Here the code supports: `'alexnet', 'vgg11', 'resnet18', 'resnet34', 'resnet50', 'vit'`

- `--init_scale`: initialization scale of the model parameter. Model initial parameters equal to `init_scale*torch.default_weight`.

- `--data`: Dataset. Here we support: `'Mnist', 'Cifar10', 'Cifar100'`.

- `--loss_fn`: loss function. We can choose to use cross-entropy or mse loss by setting it to `'ce_loss', 'mse_loss'` 
- `--optimizer`: algorithm to update model parameters. Here we support `'sgd', 'lars', 'lamb', 'adamw'`.
- `--batch_size`: batch size 
- `--num_epoch`: number of epochs 
- `--lr_setting lr_max epoch_warm lr_warm`: initial learning rate with a warm-up. If `epoch_warm>0`, then the initial learning rate would be `lr_warm` and grows linearly to `lr_max` in the first `epoch_warm` epochs.
If `epoch_warm=0`, there will be no warm-up and the initial learning rate is `lr_max`. 
- `--k_M k M`: loss augmentation for CIFAR-100 data. When training CIFAR-100, the classification target  changes `onehot` vectors to weighted target: 
target = k * `one_hot` + 1 * `ones`, and 
loss = (k * (`outputs` - M * `one_hot`)**2).`mean()` `--decay_rate`, `--decay_stepsize`: weight decay for learning rate scheduler.
- `--seed`: random seed 
- `--num_run`: the number of runs. 
- `--call_wandb`, `--no-call_wandb`: connect or don't connect with wandb.
- `--gradient`, `--no-gradient`: record gradient. 
- `--path`: saving path.
- `--log`: record detailed or essential parameters in the training process.
- `--scheduler`:  scheduler for `Vit`.

## Citation
```
@article{ma2022behind,
	title={Behind the Scenes of Gradient Descent: A Trajectory Analysis via Basis Function Decomposition},
	author={Ma, Jianhao and Guo, Lingjun and Fattahi, Salar},
	journal={arXiv preprint arXiv:2210.00346},
	year={2022}
}
```

