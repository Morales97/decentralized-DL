#!/bin/sh
python train_cifar.py --lr=3.2 --expt_name=C1.3_ring8_ring16 --topology ring ring --local_steps 0 0 --n_nodes 8 16 --start_epoch_phases 0 75 --epochs=225 --lr_decay 75 150 --dataset=cifar100 --seed=0
python train_cifar.py --lr=3.2 --expt_name=C1.3_ring8_ring16 --topology ring ring --local_steps 0 0 --n_nodes 8 16 --start_epoch_phases 0 75 --epochs=225 --lr_decay 75 150 --dataset=cifar100 --seed=1
python train_cifar.py --lr=3.2 --expt_name=C1.3_ring8_ring16 --topology ring ring --local_steps 0 0 --n_nodes 8 16 --start_epoch_phases 0 75 --epochs=225 --lr_decay 75 150 --dataset=cifar100 --seed=2

