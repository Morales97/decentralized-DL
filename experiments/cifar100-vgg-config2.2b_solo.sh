#!/bin/sh
python train_cifar.py --dataset=cifar100 --lr=0.32 --expt_name=C2.2b_solo_split --net=vgg16bn --topology solo solo solo --local_steps 0 0 0 --n_nodes 1 1 1 --data_split=True --batch_size 1024 2048 4096 --start_epoch_phases 0 66 133 --wd=5e-4 --epochs=200 --lr_decay 200 --seed=0 &
python train_cifar.py --dataset=cifar100 --lr=0.32 --expt_name=C2.2b_solo_split --net=vgg16bn --topology solo solo solo --local_steps 0 0 0 --n_nodes 1 1 1 --data_split=True --batch_size 1024 2048 4096 --start_epoch_phases 0 66 133 --wd=5e-4 --epochs=200 --lr_decay 200 --seed=1 &
wait