#!/bin/sh
python train_cifar.py --lr=3.2 --expt_name=C2.2_momentum_drop --momentum_drop=150 --topology=fully_connected  --n_nodes 8 16 32 --start_epoch_phases 0 100 200 --epochs=300 --lr_decay 300 --dataset=cifar100 --seed=0 --eval_on_average_model=True &
python train_cifar.py --lr=3.2 --expt_name=C2.2_momentum_drop --momentum_drop=150 --topology=fully_connected  --n_nodes 8 16 32 --start_epoch_phases 0 100 200 --epochs=300 --lr_decay 300 --dataset=cifar100 --seed=1 --eval_on_average_model=True &
python train_cifar.py --lr=3.2 --expt_name=C2.2_momentum_drop --momentum_drop=150 --topology=fully_connected  --n_nodes 8 16 32 --start_epoch_phases 0 100 200 --epochs=300 --lr_decay 300 --dataset=cifar100 --seed=2 --eval_on_average_model=True &
wait