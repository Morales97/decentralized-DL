#!/bin/sh
python train_cifar.py --lr=0.32 --expt_name=VGG16BN_C2.1_ring --dataset=cifar100 --net=vgg16bn --n_nodes 8 16 32 --topology ring ring ring --local_steps 0 0 0 --start_epoch_phases 0 66 133 --wd=5e-4 --epochs=200 --lr_decay 200 --eval_on_average_model=True --seed=0 &
python train_cifar.py --lr=0.32 --expt_name=VGG16BN_C2.1_ring --dataset=cifar100 --net=vgg16bn --n_nodes 8 16 32 --topology ring ring ring --local_steps 0 0 0 --start_epoch_phases 0 66 133 --wd=5e-4 --epochs=200 --lr_decay 200 --eval_on_average_model=True --seed=1 &
# python train_cifar.py --lr=0.32 --expt_name=VGG16BN_C2.1_ring --dataset=cifar100 --net=vgg16bn --n_nodes 8 16 32 --topology ring ring ring --local_steps 0 0 0 --start_epoch_phases 0 66 133 --wd=5e-4 --epochs=200 --lr_decay 200 --eval_on_average_model=True --seed=2 &
wait