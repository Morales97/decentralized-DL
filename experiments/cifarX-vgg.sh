#!/bin/sh
python train.py --dataset=cifar$1 --expt_name=val_$2 --seed=0 --lr=$2 --wd=5e-4 --epochs=200 --net=vgg16 &
python train.py --dataset=cifar$1 --expt_name=val_$2 --seed=1 --lr=$2 --wd=5e-4 --epochs=200 --net=vgg16 &
python train.py --dataset=cifar$1 --expt_name=val_$2 --seed=2 --lr=$2 --wd=5e-4 --epochs=200 --net=vgg16 &
wait