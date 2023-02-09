#!/bin/sh
python train_cifar.py --lr=3.2 --topology=ring local_steps=4 --dataset=cifar100 --seed=0 
python train_cifar.py --lr=3.2 --topology=ring local_steps=4 --dataset=cifar100 --seed=1 
python train_cifar.py --lr=3.2 --topology=ring local_steps=4 --dataset=cifar100 --seed=2
