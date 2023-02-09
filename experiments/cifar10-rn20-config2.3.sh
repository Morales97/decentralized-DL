#!/bin/sh
python train_cifar.py --lr=3.2 --expt_name=C2.3_EG_node_increase --topology exponential_graph exponential_graph exponential_graph --n_nodes 8 16 32 --start_epoch_phases 0 100 200 --epochs=300 --lr_decay 300 --seed=0 &
python train_cifar.py --lr=3.2 --expt_name=C2.3_EG_node_increase --topology exponential_graph exponential_graph exponential_graph --n_nodes 8 16 32 --start_epoch_phases 0 100 200 --epochs=300 --lr_decay 300 --seed=1 &
python train_cifar.py --lr=3.2 --expt_name=C2.3_EG_node_increase --topology exponential_graph exponential_graph exponential_graph --n_nodes 8 16 32 --start_epoch_phases 0 100 200 --epochs=300 --lr_decay 300 --seed=2 &
wait