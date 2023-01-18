#!/bin/sh
# python train_cifar.py --n_nodes=1 --lr=0.2 --topology=solo --steps_eval=400 --lr_warmup_epochs=0 --dataset=cifar100 --seed=0
python train_cifar.py --n_nodes=1 --lr=0.2 --topology=solo --steps_eval=400 --lr_warmup_epochs=0 --dataset=cifar100 --seed=0 &
python train_cifar.py --n_nodes=1 --lr=0.2 --topology=solo --steps_eval=400 --lr_warmup_epochs=0 --dataset=cifar100 --seed=1 &
python train_cifar.py --n_nodes=1 --lr=0.2 --topology=solo --steps_eval=400 --lr_warmup_epochs=0 --dataset=cifar100 --seed=2 &
python train_cifar.py --lr=3.2 --topology=ring --dataset=cifar100 --seed=0 &
python train_cifar.py --lr=3.2 --topology=ring --dataset=cifar100 --seed=1 &
python train_cifar.py --lr=3.2 --topology=ring --dataset=cifar100 --seed=2 &
python train_cifar.py --lr=3.2 --topology=fully_connected --eval_on_average_model=True --dataset=cifar100 --seed=0 &
python train_cifar.py --lr=3.2 --topology=fully_connected --eval_on_average_model=True --dataset=cifar100 --seed=1 &
python train_cifar.py --lr=3.2 --topology=fully_connected --eval_on_average_model=True --dataset=cifar100 --seed=2 &


wait
