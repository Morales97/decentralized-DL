#!/bin/sh
python train_cifar.py --lr=3.2 --expt_name=C1.4_ring8_ringlocal4 --topology ring ring --local_steps 0 4 --n_nodes 8 16 --start_epoch_phases 0 150 --dataset=cifar100 --seed=0
python train_cifar.py --lr=3.2 --expt_name=C1.4_ring8_ringlocal4 --topology ring ring --local_steps 0 4 --n_nodes 8 16 --start_epoch_phases 0 150 --dataset=cifar100 --seed=1
python train_cifar.py --lr=3.2 --expt_name=C1.4_ring8_ringlocal4 --topology ring ring --local_steps 0 4 --n_nodes 8 16 --start_epoch_phases 0 150 --dataset=cifar100 --seed=2
