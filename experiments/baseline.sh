#!/bin/sh

# Pamareters
# $1: dataset [10, 100] for [cifar10, cifar100]
# $2: network [rn20]

# baseline for rn18
python train_cifar.py --n_nodes=1 --lr=0.1 --topology=solo data_split=True --steps_eval=400 --dataset=cifar$1 --net=rn18 &
python train_cifar.py --n_nodes=1 --lr=0.2 --topology=solo data_split=True --steps_eval=400 --dataset=cifar$1 --net=rn18 &

wait