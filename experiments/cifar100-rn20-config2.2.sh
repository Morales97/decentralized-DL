#!/bin/sh
python train_cifar.py --lr=3.2 --expt_name=C2.2_FC_node_increase --topology fully_connected fully_connected fully_connected --local_steps 0 0 0 --n_nodes 8 16 32 --start_epoch_phases 0 100 200 --epochs=300 --lr_decay 300 --dataset=cifar100 --seed=0
