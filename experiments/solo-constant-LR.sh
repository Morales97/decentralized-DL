#!/bin/sh

# Pamareters
# $1: dataset [10, 100] for [cifar10, cifar100]
# $2: network [rn20]
# $3: LR

# train solo with B=1024 and constant LRs
# do keep in mind that we use LR WARMUP

python train_cifar.py --lr=$3 --expt_name=constant_lr_$3_B8 --topology=solodata_split=True --batch_size=1024 --n_nodes=1 --epochs=600 --lr_decay 600 --dataset=cifar$1 --net=$2 --seed=0 &
python train_cifar.py --lr=$3 --expt_name=constant_lr_$3_B8 --topology=solodata_split=True --batch_size=1024 --n_nodes=1 --epochs=600 --lr_decay 600 --dataset=cifar$1 --net=$2 --seed=1 &
python train_cifar.py --lr=$3 --expt_name=constant_lr_$3_B8 --topology=solodata_split=True --batch_size=1024 --n_nodes=1 --epochs=600 --lr_decay 600 --dataset=cifar$1 --net=$2 --seed=2 &

wait