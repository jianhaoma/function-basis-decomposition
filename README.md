# incremental-learning

grad_indep.py

  grad_indep.py gives figure2(b) in our paper. It captures the maximum absolute value of the inner product between \\nabla\\beta_{i}(\\theta_{t}) and \\nabla\\beta_{j}(\\theta_{t}),per 10 iterations for the first 100 iteration. We use a 3 block cnn and MNIST dataset. Simply run the file and it will produce the figure.

beta_domin.py

  beta_domin.py gives figure2(c) in our paper. It captures the log relationship between |\\beta_{i}(\\theta_{t})| and \|\\nabla\\beta_{i}\\theta_{t}\| for the first 400 iterations. We use a 3 block cnn and MNIST dataset. Simply run the file and it will produce the figure.

small-init.py

  small-init.py gives figure2(a) in our paper. It calculates the initial |\\beta_{i}(\\theta_{0})| and indexed by the coefficients of final A-CK. Please run the file to get the resutls. 

mnist.py

  mnist.py gives figure figure3(a)(b)(c) in our paper. Please change the 'num_conv_layers'(=2,3,4) parameter to get the result for different depth cnn. For 4 block cnn, please change 'seed' to 64 to get the same figure as in our paper.

train.py:

  train.py gives figure1(d)(e)(f), figure3(d)(e)(f), and figure5,6,7,8. It will return basis coefficients, train loss and test/train accuracy for given model, datasets and model parameters.
  
  running example:

  srun -l python3 train.py --call_wandb --model='alexnet' --init_scale=1 --data='Cifar10' --loss_fn='mse_loss' --optimizer='lars' --batch_size=512 --num_epoch=300 --lr_setting 2 0 1e-2 --decay_rate=0.33 --decay_stepsize=50 --seed=0 --num_run=0 --no-gradient 

    PyTorch Incremental Learning Experiments

    usage: train.py [-h] [--model MODEL] [--init_scale INIT_SCALE] [--data DATA]
                [--loss_fn LOSS_FN] [--optimizer OPTIMIZER]
                [--batch_size BATCH_SIZE] [--num_epoch NUM_EPOCH]
                [--lr_setting [LR_SETTING [LR_SETTING ...]]]
                [--decay_rate DECAY_RATE] [--decay_stepsize DECAY_STEPSIZE]
                [--seed SEED] [--num_run NUM_RUN] [--call_wandb]
                [--no-call_wandb] [--gradient] [--no-gradient] [--path PATH]
                [--log LOG] [--k_M [k M ] [--scheduler SCHEDULER]

main arguments:
  -h, --help            show this help message and exit
  --model MODEL         type of model. here the model supports: 'alexnet', 'vgg11',     'resnet18', 'resnet34', 'resnet50', 'vit'
          
  --init_scale INIT_SCALE
                        scale model initial parameter, or small initialization in our paper.

  --data DATA           type of dataset. here we support: 'Mnist', 'Cifar10', 'Cifar100'

  --loss_fn LOSS_FN     loss type. here we support: 'mse_loss', 'ce_loss'(cross entropy)
  --optimizer OPTIMIZER
                        optimizer type. here we support: 'sgd', 'lars', 'lamb', 'adamw'
  --batch_size BATCH_SIZE
                        batch size
  --num_epoch NUM_EPOCH
                        number of epoch
  --lr_setting [LR_SETTING [LR_SETTING ...]]
                        settings for tuning learning rate: [max_learning rate,
                        warm_up epochs, initial_lr]. if warm_up epochs=0, there is no warm up.
  --k_M [K_M [K_M ...]]
                        loss augumenting for cifar-100 data: weight = k *
                        one_hot + 1 * ones, loss = (weight * (outputs - M *
                        one_hot)**2).mean()
other arguments:
  --decay_rate DECAY_RATE
                        the decay rate in each stepsize decay
  --decay_stepsize DECAY_STEPSIZE
                        the decay time
  --seed SEED           random seed
  --num_run NUM_RUN     the number of run
  --call_wandb          connect with wandb or not
  --no-call_wandb
  --gradient            document changes with gradient
  --no-gradient
  --path PATH           saving path
  --log LOG             how much we want to document
  --scheduler SCHEDULER
                        what scheduler for vit

