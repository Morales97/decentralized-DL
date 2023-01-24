#!/bin/sh
python train_cifar.py --expt_name=C2.9_n16 --topology=fully_connected --eval_on_average_model=True --lr=3.2 --lr_decay=300 --seed=0 &
python train_cifar.py --expt_name=C2.9_n16 --topology=fully_connected --eval_on_average_model=True --lr=3.2 --lr_decay=300 --seed=1 &
python train_cifar.py --expt_name=C2.9_n16 --topology=fully_connected --eval_on_average_model=True --lr=3.2 --lr_decay=300 --seed=2 &
wait