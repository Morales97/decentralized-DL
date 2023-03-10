#!/bin/sh
python train.py --dataset=cifar$1 --expt_name=val_$2 --seed=0 --lr=$2 --wd=5e-4 --epochs=200 --alpha 0.968 0.984 0.992 0.996 0.998 --net=vgg16 --ema_period=16 --avg_index &
python train.py --dataset=cifar$1 --expt_name=val_$2 --seed=1 --lr=$2 --wd=5e-4 --epochs=200 --alpha 0.968 0.984 0.992 0.996 0.998 --net=vgg16 --ema_period=16 --avg_index &
python train.py --dataset=cifar$1 --expt_name=val_$2 --seed=2 --lr=$2 --wd=5e-4 --epochs=200 --alpha 0.968 0.984 0.992 0.996 0.998 --net=vgg16 --ema_period=16 --avg_index &
wait