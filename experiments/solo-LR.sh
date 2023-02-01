#!/bin/sh

# Pamareters
# $1: dataset [cifar10, cifar100]
# $2: network [rn20]

# train solo with B=128 and different LRs
# do keep in mind that we use LR WARMUP

#python train_cifar.py --n_nodes=1 --lr=0.2 --topology=solo --steps_eval=400 --dataset=cifar$1 --net=$2 &
python train_cifar.py --n_nodes=1 --lr=0.4 --topology=solo --steps_eval=400 --dataset=cifar$1 --net=$2 &
python train_cifar.py --n_nodes=1 --lr=0.8 --topology=solo --steps_eval=400 --dataset=cifar$1 --net=$2 &
python train_cifar.py --n_nodes=1 --lr=1.6 --topology=solo --steps_eval=400 --dataset=cifar$1 --net=$2 &
wait