#!/bin/sh
python train_cifar.py --lr=3.2 --topology=fully_connected eval_on_average_model=True --expt_name=FC_std0.1 --model_std=0.1 --dataset=cifar100 --seed=0 &
python train_cifar.py --lr=3.2 --topology=fully_connected eval_on_average_model=True --expt_name=FC_std0.1 --model_std=0.1 --dataset=cifar100 --seed=1 &
python train_cifar.py --lr=3.2 --topology=fully_connected eval_on_average_model=True --expt_name=FC_std0.01 --model_std=0.01 --dataset=cifar100 --seed=0 &
python train_cifar.py --lr=3.2 --topology=fully_connected eval_on_average_model=True --expt_name=FC_std0.01 --model_std=0.01 --dataset=cifar100 --seed=1 &
python train_cifar.py --lr=3.2 --topology=fully_connected eval_on_average_model=True --expt_name=FC_std0.001 --model_std=0.001 --dataset=cifar100 --seed=0 &
python train_cifar.py --lr=3.2 --topology=fully_connected eval_on_average_model=True --expt_name=FC_std0.001 --model_std=0.001 --dataset=cifar100 --seed=1 &
python train_cifar.py --lr=3.2 --topology=fully_connected eval_on_average_model=True --expt_name=FC_std0.0001 --model_std=0.0001 --dataset=cifar100 --seed=0 &
python train_cifar.py --lr=3.2 --topology=fully_connected eval_on_average_model=True --expt_name=FC_std0.0001 --model_std=0.0001 --dataset=cifar100 --seed=1 &

wait
