#!/bin/sh
# train sequentially (no &)
python train.py --expt_name=finetune_val_SGD_0.1 --epochs=50 --lr=0.1 --lr_warmup_epochs=0 --seed=0 --dataset=cifar100 --net=rn18 --pretrained=/mloraw1/danmoral/checkpoints/tiny-in/rn18/val_0.8_s$1/checkpoint_last.pth.tar 
python train.py --expt_name=finetune_val_SGD_0.3 --epochs=50 --lr=0.3 --lr_warmup_epochs=0 --seed=0 --dataset=cifar100 --net=rn18 --pretrained=/mloraw1/danmoral/checkpoints/tiny-in/rn18/val_0.8_s$1/checkpoint_last.pth.tar 
python train.py --expt_name=finetune_val_SGD_0.03 --epochs=50 --lr=0.03 --lr_warmup_epochs=0 --seed=0 --dataset=cifar100 --net=rn18 --pretrained=/mloraw1/danmoral/checkpoints/tiny-in/rn18/val_0.8_s$1/checkpoint_last.pth.tar 
python train.py --expt_name=finetune_val_SGD_0.01 --epochs=50 --lr=0.01 --lr_warmup_epochs=0 --seed=0 --dataset=cifar100 --net=rn18 --pretrained=/mloraw1/danmoral/checkpoints/tiny-in/rn18/val_0.8_s$1/checkpoint_last.pth.tar 
wait