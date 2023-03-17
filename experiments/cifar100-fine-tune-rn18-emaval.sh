#!/bin/sh
# train sequentially (no &)
python train.py --expt_name=finetune_val_EMAacc_0.1 --epochs=50 --lr=0.1 --lr_warmup_epochs=0 --seed=0 --alpha 0.984 0.992 0.996 --dataset=cifar100 --net=rn18 --pretrained=/mloraw1/danmoral/checkpoints/tiny-in/rn18/val_0.8_s$1/best_ema_loss.pth.tar 
python train.py --expt_name=finetune_val_EMAacc_0.3 --epochs=50 --lr=0.3 --lr_warmup_epochs=0 --seed=0 --alpha 0.984 0.992 0.996 --dataset=cifar100 --net=rn18 --pretrained=/mloraw1/danmoral/checkpoints/tiny-in/rn18/val_0.8_s$1/best_ema_loss.pth.tar 
python train.py --expt_name=finetune_val_EMAacc_0.03 --epochs=50 --lr=0.03 --lr_warmup_epochs=0 --seed=0 --alpha 0.984 0.992 0.996 --dataset=cifar100 --net=rn18 --pretrained=/mloraw1/danmoral/checkpoints/tiny-in/rn18/val_0.8_s$1/best_ema_loss.pth.tar 
python train.py --expt_name=finetune_val_EMAacc_0.01 --epochs=50 --lr=0.01 --lr_warmup_epochs=0 --seed=0 --alpha 0.984 0.992 0.996 --dataset=cifar100 --net=rn18 --pretrained=/mloraw1/danmoral/checkpoints/tiny-in/rn18/val_0.8_s$1/best_ema_loss.pth.tar 
wait