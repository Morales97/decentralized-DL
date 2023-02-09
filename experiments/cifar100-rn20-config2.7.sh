#!/bin/sh
python train_cifar.py --lr=3.2 --expt_name=C2.7_FC_n8 --topology=fully_connected  --n_nodes 8 8 8 --lr 3.2 1.6 0.8 --start_epoch_phases 0 100 200 --epochs=300 --lr_decay 300 --dataset=cifar100 --seed=0 --eval_on_average_model=True &
python train_cifar.py --lr=3.2 --expt_name=C2.7_FC_n8 --topology=fully_connected  --n_nodes 8 8 8 --lr 3.2 1.6 0.8 --start_epoch_phases 0 100 200 --epochs=300 --lr_decay 300 --dataset=cifar100 --seed=1 --eval_on_average_model=True &
python train_cifar.py --lr=3.2 --expt_name=C2.7_FC_n8 --topology=fully_connected  --n_nodes 8 8 8 --lr 3.2 1.6 0.8 --start_epoch_phases 0 100 200 --epochs=300 --lr_decay 300 --dataset=cifar100 --seed=2 --eval_on_average_model=True &
wait