#!/bin/sh

# CIFAR 10
python train_cifar.py --init_momentum=False --lr=1.6 --expt_name=C2.8_no_init_mom --topology=fully_connected  --n_nodes 8 16 32 --start_epoch_phases 0 100 200 --epochs=300 --lr_decay 300 --seed=0 --eval_on_average_model=True &
python train_cifar.py --init_momentum=False --lr=1.6 --expt_name=C2.8_no_init_mom --topology=fully_connected  --n_nodes 8 16 32 --start_epoch_phases 0 100 200 --epochs=300 --lr_decay 300 --seed=1 --eval_on_average_model=True &
python train_cifar.py --init_momentum=False --lr=1.6 --expt_name=C2.8_no_init_mom --topology=fully_connected  --n_nodes 8 16 32 --start_epoch_phases 0 100 200 --epochs=300 --lr_decay 300 --seed=2 --eval_on_average_model=True &

# CIFAR 100
# python train_cifar.py --lr=1.6 --expt_name=C2.8_FC_node_increase --topology=fully_connected  --n_nodes 8 16 32 --start_epoch_phases 0 100 200 --epochs=300 --lr_decay 300 --dataset=cifar100 --seed=0 --eval_on_average_model=True &
# python train_cifar.py --lr=1.6 --expt_name=C2.8_FC_node_increase --topology=fully_connected  --n_nodes 8 16 32 --start_epoch_phases 0 100 200 --epochs=300 --lr_decay 300 --dataset=cifar100 --seed=1 --eval_on_average_model=True &
# python train_cifar.py --lr=1.6 --expt_name=C2.8_FC_node_increase --topology=fully_connected  --n_nodes 8 16 32 --start_epoch_phases 0 100 200 --epochs=300 --lr_decay 300 --dataset=cifar100 --seed=2 --eval_on_average_model=True &
wait