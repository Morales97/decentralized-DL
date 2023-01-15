#!/bin/sh
python train_cifar.py --lr=3.2 --topology ring fully_connected --local_steps 0 16 --n_nodes 16 16 --start_epoch_phases 0 150 --dataset=cifar100 --seed=0
python train_cifar.py --lr=3.2 --topology ring fully_connected --local_steps 0 16 --n_nodes 16 16 --start_epoch_phases 0 150 --dataset=cifar100 --seed=1
python train_cifar.py --lr=3.2 --topology ring fully_connected --local_steps 0 16 --n_nodes 16 16 --start_epoch_phases 0 150 --dataset=cifar100 --seed=2
