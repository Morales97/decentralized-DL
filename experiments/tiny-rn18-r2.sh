#!/bin/sh


# SGD run
python train.py --dataset=tiny-in --expt_name=SGD_$2 --seed=$1 --lr=$2 --val_fraction=0 --net=rn18 --avg_index --epochs=150 --ema_acc_epoch=$3 --ema_val_epoch=$4 # TODO get epochs automatically from ema ckpts 

# EMA acc
python train.py --dataset=tiny-in --expt_name=EMA_acc_$2 --seed=$1 --val_fraction=0 --net=rn18 --avg_index --epochs=150 --lr_decay=constant --epoch_swa=$3 --resume=/mloraw1/danmoral/checkpoints/tiny-in/rn18/SGD_$2_s$1/ema_acc_epoch.pth.tar

# EMA val
python train.py --dataset=tiny-in --expt_name=EMA_val_$2 --seed=$1 --val_fraction=0 --net=rn18 --avg_index --epochs=150 --lr_decay=constant --epoch_swa=$4 --resume=/mloraw1/danmoral/checkpoints/tiny-in/rn18/SGD_$2_s$1/ema_val_epoch.pth.tar

