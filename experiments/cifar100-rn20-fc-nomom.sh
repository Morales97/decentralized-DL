#!/bin/sh
python train_cifar.py --expt_name=FC_16_NOmom --topology=fully_connected eval_on_average_model=True --steps_eval=100 --dataset=cifar100 --momentum=0 --nesterov=False --seed=0 &
python train_cifar.py --expt_name=FC_16_NOmom --topology=fully_connected eval_on_average_model=True --steps_eval=100 --dataset=cifar100 --momentum=0 --nesterov=False --seed=1 &
python train_cifar.py --expt_name=FC_16_NOmom --topology=fully_connected eval_on_average_model=True --steps_eval=100 --dataset=cifar100 --momentum=0 --nesterov=False --seed=2 &
wait
