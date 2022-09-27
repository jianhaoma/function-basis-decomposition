#!/bin/bash
#SBATCH --job-name train
#SBATCH --nodes=1
#SBATCH --time=03:59:59
#SBATCH --account=user_account
#SBATCH --mail-user=user@mail
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=20G
source file_path
srun -l python3 train.py --call_wandb --model='fc-elu' --init_scale=1 --data='Cifar100' --loss_fn='mse_loss' --optimizer='sgd' --batch_size=24 --num_epoch=1 --lr_setting 1e-1 5 1e-2 --decay_rate=0.33 --decay_stepsize=50 --seed=0 --num_run=0 --no-gradient 
echo "Finished!"
