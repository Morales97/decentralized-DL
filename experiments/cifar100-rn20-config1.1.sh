#!/bin/sh
python train_cifar.py --lr=3.2 --expt_name=C1.1_ring8_local16 --topology ring fully_connected local_steps 0 16 --n_nodes 8 16 --start_epoch_phases 0 150 --dataset=cifar100 --seed=0
python train_cifar.py --lr=3.2 --expt_name=C1.1_ring8_local16 --topology ring fully_connected local_steps 0 16 --n_nodes 8 16 --start_epoch_phases 0 150 --dataset=cifar100 --seed=1
python train_cifar.py --lr=3.2 --expt_name=C1.1_ring8_local16 --topology ring fully_connected local_steps 0 16 --n_nodes 8 16 --start_epoch_phases 0 150 --dataset=cifar100 --seed=2