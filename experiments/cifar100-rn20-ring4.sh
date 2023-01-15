#!/bin/sh
python train_cifar.py --n_nodes=4 --epochs=75 --lr=3.2 --topology=ring --dataset=cifar100 --seed=0
python train_cifar.py --n_nodes=4 --epochs=75 --lr=3.2 --topology=ring --dataset=cifar100 --seed=1
python train_cifar.py --n_nodes=4 --epochs=75 --lr=3.2 --topology=ring --dataset=cifar100 --seed=2
