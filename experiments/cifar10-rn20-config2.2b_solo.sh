#!/bin/sh
python train_cifar.py --lr=3.2 --expt_name=C2.2b_solo_split --topology solo solo solo --local_steps 0 0 0 --n_nodes 1 1 1 --data_split=True --batch_size 1024 2048 4096 --start_epoch_phases 0 100 200 --epochs=300 --lr_decay 300 --seed=0 &
python train_cifar.py --lr=3.2 --expt_name=C2.2b_solo_split  --topology solo solo solo --local_steps 0 0 0 --n_nodes 1 1 1 --data_split=True --batch_size 1024 2048 4096 --start_epoch_phases 0 100 200 --epochs=300 --lr_decay 300 --seed=1 &
python train_cifar.py --lr=3.2 --expt_name=C2.2b_solo_split --topology solo solo solo --local_steps 0 0 0 --n_nodes 1 1 1 --data_split=True --batch_size 1024 2048 4096 --start_epoch_phases 0 100 200 --epochs=300 --lr_decay 300 --seed=2 &
wait