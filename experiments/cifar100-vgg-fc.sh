#!/bin/sh

# Actual 16 nodes
# python train_cifar.py --expt_name=VGG16BN_FC_16_lr0.32 --dataset=cifar100 --net=vgg16bn --n_nodes=16 --topology=fully_connected --lr=0.32 --wd=5e-4 --epochs=200 --lr_decay 100 175 --seed=1 --eval_on_average_model=True &
# python train_cifar.py --expt_name=VGG16BN_FC_16_lr0.32 --dataset=cifar100 --net=vgg16bn --n_nodes=16 --topology=fully_connected --lr=0.32 --wd=5e-4 --epochs=200 --lr_decay 100 175 --seed=2 --eval_on_average_model=True &

wait