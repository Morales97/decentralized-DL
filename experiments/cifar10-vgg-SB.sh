#!/bin/sh
#python train_cifar.py --n_nodes=1 --lr=0.05 --topology=solo --steps_eval=400 --net=vgg11 --epochs=100 --lr_warmup_epochs=0

# ring
python train_cifar.py --n_nodes=16 --lr=0.8 --topology=ring --steps_eval=400 --net=vgg11 --epochs=100

# FC
python train_cifar.py --n_nodes=16 --lr=0.8 --topology=fully_connected --eval_on_average_model=True --steps_eval=400 --net=vgg11 --epochs=100
