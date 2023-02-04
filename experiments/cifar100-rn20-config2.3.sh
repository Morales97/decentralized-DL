#!/bin/sh
# python train_cifar.py --lr=3.2 --expt_name=C2.3_EG_node_increase --topology=exponential_graph --n_nodes 8 16 32 --start_epoch_phases 0 100 200 --epochs=300 --lr_decay 300 --dataset=cifar100 --seed=0 &
# python train_cifar.py --lr=3.2 --expt_name=C2.3_EG_node_increase --topology=exponential_graph --n_nodes 8 16 32 --start_epoch_phases 0 100 200 --epochs=300 --lr_decay 300 --dataset=cifar100 --seed=1 &
# python train_cifar.py --lr=3.2 --expt_name=C2.3_EG_node_increase --topology=exponential_graph --n_nodes 8 16 32 --start_epoch_phases 0 100 200 --epochs=300 --lr_decay 300 --dataset=cifar100 --seed=2 &

# python train_cifar.py --lr=3.2 --expt_name=C2.3_EG_8n --topology=exponential_graph --n_nodes=8 --epochs=300 --lr_decay 100 200 --lr_decay_factor=2 --dataset=cifar100 --seed=0 &
# python train_cifar.py --lr=3.2 --expt_name=C2.3_EG_8n --topology=exponential_graph --n_nodes=8 --epochs=300 --lr_decay 100 200 --lr_decay_factor=2 --dataset=cifar100 --seed=1 &
# python train_cifar.py --expt_name=C2.3_EG_mixed_16 --topology exponential_graph exponential_graph fully_connected --n_nodes=16 --epochs=300 --lr_decay=300 --start_epoch_phases 0 100 200 --lr 3.2 1.6 0.8 --dataset=cifar100 --seed=0 &
# python train_cifar.py --expt_name=C2.3_EG_mixed_16 --topology exponential_graph exponential_graph fully_connected --n_nodes=16 --epochs=300 --lr_decay=300 --start_epoch_phases 0 100 200 --lr 3.2 1.6 0.8 --dataset=cifar100 --seed=1 &
# python train_cifar.py --expt_name=C2.3_EG_mixed_16 --topology exponential_graph exponential_graph fully_connected --n_nodes=16 --epochs=300 --lr_decay=300 --start_epoch_phases 0 100 200 --lr 3.2 1.6 0.8 --dataset=cifar100 --seed=2 &

# alphas
python train_cifar.py --expt_name=C2.3_EG_mixed_$1_mEMA --alpha 0.984 0.992 0.996 0.998 0.999 0.9995 --topology exponential_graph exponential_graph fully_connected --n_nodes=$1 --epochs=300 --lr_decay=300 --start_epoch_phases 0 100 200 --lr 3.2 1.6 0.8 --dataset=cifar100 --seed=0 &
python train_cifar.py --expt_name=C2.3_EG_mixed_$1_mEMA --alpha 0.984 0.992 0.996 0.998 0.999 0.9995 --topology exponential_graph exponential_graph fully_connected --n_nodes=$1 --epochs=300 --lr_decay=300 --start_epoch_phases 0 100 200 --lr 3.2 1.6 0.8 --dataset=cifar100 --seed=1 &
python train_cifar.py --expt_name=C2.3_EG_mixed_$1_mEMA --alpha 0.984 0.992 0.996 0.998 0.999 0.9995 --topology exponential_graph exponential_graph fully_connected --n_nodes=$1 --epochs=300 --lr_decay=300 --start_epoch_phases 0 100 200 --lr 3.2 1.6 0.8 --dataset=cifar100 --seed=2 &

wait