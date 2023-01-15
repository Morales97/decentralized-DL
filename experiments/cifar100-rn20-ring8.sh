#!/bin/sh
python train_cifar.py --n_nodes=8 --epochs=150 --lr=3.2 --topology=ring --dataset=cifar100 --seed=0
python train_cifar.py --n_nodes=8 --epochs=150 --lr=3.2 --topology=ring --dataset=cifar100 --seed=1
python train_cifar.py --n_nodes=8 --epochs=150 --lr=3.2 --topology=ring --dataset=cifar100 --seed=2