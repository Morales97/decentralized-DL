#!/bin/sh
python train_cifar.py --lr=3.2 --expt_name=C2.7_solotopology solo solo solo --n_nodes 1 1 1 --batch_size 1024 1024 1024 --lr 3.2 1.6 0.8 --start_epoch_phases 0 100 200 --epochs=300 --lr_decay 300 --dataset=cifar$1 --seed=0 &
python train_cifar.py --lr=3.2 --expt_name=C2.7_solotopology solo solo solo --n_nodes 1 1 1 --batch_size 1024 1024 1024 --lr 3.2 1.6 0.8 --start_epoch_phases 0 100 200 --epochs=300 --lr_decay 300 --dataset=cifar$1 --seed=1 &
python train_cifar.py --lr=3.2 --expt_name=C2.7_solotopology solo solo solo --n_nodes 1 1 1 --batch_size 1024 1024 1024 --lr 3.2 1.6 0.8 --start_epoch_phases 0 100 200 --epochs=300 --lr_decay 300 --dataset=cifar$1 --seed=2 &
wait