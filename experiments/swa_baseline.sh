#!/bin/sh

# Pamareters
# $1: dataset [10, 100] for [cifar10, cifar100]
# $2: network [rn20]

# baseline for rn18
python train_cifar.py --expt_name=SWA_0.1_0.05 --n_nodes=1 --lr=0.1 --epochs=225 --epoch_swa=126 --epoch_swa_budget=150 --swa_lr=0.05 --lr_decay=226 --alpha=0.999 --topology=solo --data_split=True --steps_eval=400 --dataset=cifar100 --net=rn18 &
python train_cifar.py --expt_name=SWA_0.8_0.2 --n_nodes=1 --lr=0.8 --epochs=225 --epoch_swa=126 --epoch_swa_budget=150 --swa_lr=0.2 --lr_decay=226 --alpha=0.999 --topology=solo --data_split=True --steps_eval=400 --dataset=cifar100 --net=rn18 &
python train_cifar.py --expt_name=SWA_0.2 --n_nodes=1 --lr=0.2 --epochs=225 --epoch_swa=126 --epoch_swa_budget=150 --swa_lr=0.2 --lr_decay=226 --alpha=0.999 --topology=solo --data_split=True --steps_eval=400 --dataset=cifar100 --net=rn18 &
# python train_cifar.py --n_nodes=1 --lr=0.2 --topology=solo --data_split=True --steps_eval=400 --dataset=cifar$1 --net=rn18 &
# python train_cifar.py --n_nodes=1 --lr=0.1 --lr_decay 60 120 160 --epochs=200 --lr_decay_factor=5 --topology=solo --data_split=True --steps_eval=400 --dataset=cifar100 --net=rn18 &


# python train_cifar.py --wandb=False --data_fraction=0.01 --n_nodes=1 --lr=0.1 --epochs=20 --epoch_swa=3 --epoch_swa_budget=10 --swa_lr=0.05 --lr_decay=226 --topology=solo --steps_eval=400 --dataset=cifar100 --net=rn18

wait