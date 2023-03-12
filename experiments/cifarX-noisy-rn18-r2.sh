#!/bin/sh


# SGD run
python train.py --dataset=cifar$1 --label_noise --expt_name=SGD_$3_noise40 --seed=$2 --lr=$3 --val_fraction=0 --net=rn18 --avg_index --ema_acc_epoch=$4 --ema_val_epoch=$5 # TODO get epochs automatically from ema ckpts 

# EMA acc
python train.py --dataset=cifar$1 --label_noise --expt_name=EMA_acc_$3_noise40 --seed=$2 --val_fraction=0 --net=rn18 --avg_index --lr_decay=constant --epoch_swa=$4 --resume=/mloraw1/danmoral/checkpoints/cifar100/rn18/SGD_$3_noise40_s$2/ema_acc_epoch.pth.tar

# EMA val
python train.py --dataset=cifar$1 --label_noise --expt_name=EMA_val_$3_noise40 --seed=$2 --val_fraction=0 --net=rn18 --avg_index --lr_decay=constant --epoch_swa=$5 --resume=/mloraw1/danmoral/checkpoints/cifar100/rn18/SGD_$3_noise40_s$2/ema_val_epoch.pth.tar

