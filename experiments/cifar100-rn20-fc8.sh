#!/bin/sh
python train_cifar.py --n_nodes=8 --lr=3.2 --topology=fully_connected --eval_on_average_model=True --steps_eval=100 --dataset=cifar100 --seed=1
python train_cifar.py --n_nodes=8 --lr=3.2 --topology=fully_connected --eval_on_average_model=True --steps_eval=100 --dataset=cifar100 --seed=2
