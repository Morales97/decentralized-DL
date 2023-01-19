#!/bin/sh
# POST local 16
python train_cifar.py --lr=3.2 --expt_name=POSTlocal16 --topology fully_connected fully_connected --local_steps 0 16 --n_nodes 16 16 --start_epoch_phases 0 150 --dataset=cifar100 --seed=0 &
python train_cifar.py --lr=3.2 --expt_name=POSTlocal16 --topology fully_connected fully_connected --local_steps 0 16 --n_nodes 16 16 --start_epoch_phases 0 150 --dataset=cifar100 --seed=1 &
python train_cifar.py --lr=3.2 --expt_name=POSTlocal16 --topology fully_connected fully_connected --local_steps 0 16 --n_nodes 16 16 --start_epoch_phases 0 150 --dataset=cifar100 --seed=2 &

# POST ring local 4
python train_cifar.py --lr=3.2 --expt_name=POST_ring_local4 --topology fully_connected ring --local_steps 0 4 --n_nodes 16 16 --start_epoch_phases 0 150 --dataset=cifar100 --seed=0 &
python train_cifar.py --lr=3.2 --expt_name=POST_ring_local4 --topology fully_connected ring --local_steps 0 4 --n_nodes 16 16 --start_epoch_phases 0 150 --dataset=cifar100 --seed=1 &
python train_cifar.py --lr=3.2 --expt_name=POST_ring_local4 --topology fully_connected ring --local_steps 0 4 --n_nodes 16 16 --start_epoch_phases 0 150 --dataset=cifar100 --seed=2 &

# ring n8
python train_cifar.py --n_nodes=8 --epochs=150 --lr=3.2 --topology=ring --dataset=cifar100 --seed=0 &
python train_cifar.py --n_nodes=8 --epochs=150 --lr=3.2 --topology=ring --dataset=cifar100 --seed=1 &
python train_cifar.py --n_nodes=8 --epochs=150 --lr=3.2 --topology=ring --dataset=cifar100 --seed=2 &

wait