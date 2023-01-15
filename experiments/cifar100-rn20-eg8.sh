#!/bin/sh
python train_cifar.py --n_nodes=8 --lr=3.2 --topology=exponential_graph --dataset=cifar100 --seed=1
python train_cifar.py --n_nodes=8 --lr=3.2 --topology=exponential_graph --dataset=cifar100 --seed=2