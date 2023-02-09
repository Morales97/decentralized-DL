#!/bin/sh
python train_cifar.py --expt_name=C2.7b_600e --topology=solo  --data_split=True --batch_size=1024  --n_nodes=1 --lr 3.2 1.6 0.8 0.4 0.2 0.1 --start_epoch_phases 0 100 200 300 400 500 --epochs=600 --lr_decay 600 --dataset=cifar$1 --seed=0 &
python train_cifar.py --expt_name=C2.7b_600e --topology=solo  --data_split=True --batch_size=1024  --n_nodes=1 --lr 3.2 1.6 0.8 0.4 0.2 0.1 --start_epoch_phases 0 100 200 300 400 500 --epochs=600 --lr_decay 600 --dataset=cifar$1 --seed=1 &
python train_cifar.py --expt_name=C2.7b_600e --topology=solo  --data_split=True --batch_size=1024  --n_nodes=1 --lr 3.2 1.6 0.8 0.4 0.2 0.1 --start_epoch_phases 0 100 200 300 400 500 --epochs=600 --lr_decay 600 --dataset=cifar$1 --seed=2 &
wait