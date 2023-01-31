#!/bin/sh
python train_cifar.py --lr=3.2 --expt_name=C2.2c_500e --topology solo solo solo solo solo --data_split=True --local_steps 0 0 0 0 0 --n_nodes 1 1 1 1 1 --start_epoch_phases 0 100 200 300 400 --batch_size 1024 2048 4096 8192 16384 --epochs=500 --lr_decay 500 --dataset=cifar$1 --seed=0 &
python train_cifar.py --lr=3.2 --expt_name=C2.2c_500e --topology solo solo solo solo solo --data_split=True --local_steps 0 0 0 0 0 --n_nodes 1 1 1 1 1 --start_epoch_phases 0 100 200 300 400 --batch_size 1024 2048 4096 8192 16384 --epochs=500 --lr_decay 500 --dataset=cifar$1 --seed=1 &
wait