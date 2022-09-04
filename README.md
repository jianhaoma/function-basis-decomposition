# incremental-learning

Instruction for train.py:

  train.py will return basis coefficients, train loss and test/train accuracy for given model, datasets and model parameters.
  
  Simply type python train.py --param=given_param to run the file.

    PyTorch Incremental Learning Experiments

    optional arguments:
      -h, --help            show this help message and exit
      --model MODEL         type of model: 
                            support: fc-act(act=elu,relu,tanh,hardtanh,softmax), 
                                           cnn(activation:elu,relu,tanh,hardtanh,softmax; 
                                               batch normalization: true or false;pool: maxpool, averagepool)
                                           vgg11, resnet18, alexnet, wide-resnet, vit, swimnet.
      --init_scale INIT_SCALE
                            scale model initial parameter
      --data DATA           type of dataset: support: Mnist, Cifar10, Cifar100, Imagenet
      --loss_fn LOSS_FN     loss type
      --optimizer OPTIMIZER
                            optimizer type: support sgd(lars),adam(torch.nn)
      --batch_size BATCH_SIZE
                            batch size
      --num_epoch NUM_EPOCH
                            number of epoch
      --lr_setting [LR_SETTING [LR_SETTING ...]]
                            settings for tuning learning rate: [max_lr,
                            warm_up_epochs, initial_lr] 
                            note: default warm up is set as linearly increasing lr from lnitial_lr to max_lr in warm_up_epochs time.
                            When warm_up_epochs=0, there is no warm up process.
      --decay_rate DECAY_RATE
                            the decay rate in each stepsize decay
      --decay_stepsize DECAY_STEPSIZE
                            the decay time
      --seed SEED           random seed
      --num_run NUM_RUN     the number of run
      --call_wandb          connect with wandb 
      --no-call_wandb       don't connect with wandb
      --gradient            compute and document changes with gradient
                            note: it may cosume large cuda memory.
      --no-gradient         don't compute gradient
      --path PATH           saving path
