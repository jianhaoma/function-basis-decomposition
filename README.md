# incremental-learning

Instruction for train.py:
  train.py will return basis coefficients, train loss and test/train accuracy for given model, datasets and model parameters.
  Simply type python train.py --param=given_param to run the file.

    PyTorch Incremental Learning Experiments

    optional arguments:
      -h, --help            show this help message and exit
      --model MODEL         type of model: support: fc-act(act=elu,relu,tanh,hardtanh,softmax), vgg11, resnet18, alexnet, wide-resnet, vit.
      --init_scale INIT_SCALE
                            scale model initial parameter
      --data DATA           type of dataset: support: Cifar10, Cifar100, Imagenet
      --loss_fn LOSS_FN     loss type
      --optimizer OPTIMIZER
                            optimizer type: support sgd(lars),adam(torch.nn)
      --batch_size BATCH_SIZE
                            batch size
      --num_epoch NUM_EPOCH
                            number of epoch
      --lr_setting [LR_SETTING [LR_SETTING ...]]
                            settings for tuning learning rate: [max_learning rate,
                            warm_up epochs, initial_lr]
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
