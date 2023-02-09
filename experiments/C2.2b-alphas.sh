#!/bin/sh

# 300 epochs
# python train_cifar.py --expt_name=C2.2b_alphas --alpha 0.999 0.998 0.996 0.992 0.984 0.968 --topology=solo --n_nodes=1 --data_split=True --batch_size=1024 --lr 3.2 1.6 0.8 --start_epoch_phases 0 100 200 --epochs=300 --lr_decay 300 --dataset=cifar$1 --net=$2 --seed=0 &
# python train_cifar.py --expt_name=C2.2b_alphas --alpha 0.999 0.998 0.996 0.992 0.984 0.968 --topology=solo --n_nodes=1 --data_split=True --batch_size=1024 --lr 3.2 1.6 0.8 --start_epoch_phases 0 100 200 --epochs=300 --lr_decay 300 --dataset=cifar$1 --net=$2 --seed=1 &
# python train_cifar.py --expt_name=C2.2b_alphas --alpha 0.999 0.998 0.996 0.992 0.984 0.968 --topology=solo --n_nodes=1 --data_split=True --batch_size=1024 --lr 3.2 1.6 0.8 --start_epoch_phases 0 100 200 --epochs=300 --lr_decay 300 --dataset=cifar$1 --net=$2 --seed=2 &

# 600 epochs
python train_cifar.py --expt_name=C2.2b_alphas_600e --alpha 0.999 0.998 0.996 0.992 0.984 0.968 --topology=solo --n_nodes=1 --data_split=True --batch_size=1024 --lr 3.2 1.6 0.8 0.4 0.2 0.1 --start_epoch_phases 0 100 200 300 400 500 --epochs=600 --lr_decay 600 --dataset=cifar$1 --net=$2 --seed=0 &
python train_cifar.py --expt_name=C2.2b_alphas_600e --alpha 0.999 0.998 0.996 0.992 0.984 0.968 --topology=solo --n_nodes=1 --data_split=True --batch_size=1024 --lr 3.2 1.6 0.8 0.4 0.2 0.1 --start_epoch_phases 0 100 200 300 400 500 --epochs=600 --lr_decay 600 --dataset=cifar$1 --net=$2 --seed=1 &
python train_cifar.py --expt_name=C2.2b_alphas_600e --alpha 0.999 0.998 0.996 0.992 0.984 0.968 --topology=solo --n_nodes=1 --data_split=True --batch_size=1024 --lr 3.2 1.6 0.8 0.4 0.2 0.1 --start_epoch_phases 0 100 200 300 400 500 --epochs=600 --lr_decay 600 --dataset=cifar$1 --net=$2 --seed=2 &
wait