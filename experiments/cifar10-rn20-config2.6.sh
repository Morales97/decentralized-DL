#!/bin/sh
python train_cifar.py --lr=3.2 --expt_name=C2.6_n8 --topology=fully_connected local_steps=0 --n_nodes=8 --epochs=175 --lr_decay=300 --seed=0 --eval_on_average_model=True &
python train_cifar.py --lr=3.2 --expt_name=C2.6_n8 --topology=fully_connected local_steps=0 --n_nodes=8 --epochs=175 --lr_decay=300 --seed=1 --eval_on_average_model=True &
python train_cifar.py --lr=3.2 --expt_name=C2.6_n8 --topology=fully_connected local_steps=0 --n_nodes=8 --epochs=175 --lr_decay=300 --seed=2 --eval_on_average_model=True &
wait