#!/bin/sh

# need to reduce number of epochs (in reference to n_nodes=16) to keep same total number of steps
python train_cifar.py --n_nodes=8 --epochs=150 --lr=3.2 --topology=fully_connected eval_on_average_model=True --steps_eval=100 --dataset=cifar100 --seed=0
python train_cifar.py --n_nodes=8 --epochs=150 --lr=3.2 --topology=fully_connected eval_on_average_model=True --steps_eval=100 --dataset=cifar100 --seed=1
python train_cifar.py --n_nodes=8 --epochs=150 --lr=3.2 --topology=fully_connected eval_on_average_model=True --steps_eval=100 --dataset=cifar100 --seed=2
