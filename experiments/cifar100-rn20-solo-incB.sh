#!/bin/sh
# solo with increase of Batch size (and constant lr)
python train_cifar.py --expt_name=solo_B_increase --n_nodes 1 2 4 --lr=0.2 --lr_decay=300 --topology=fully_connected start_epoch_phases 0 100 200 --eval_on_average_model=True --steps_eval=400 --lr_warmup_epochs=0 --dataset=cifar100 --seed=0 &
python train_cifar.py --expt_name=solo_B_increase --n_nodes 1 2 4 --lr=0.2 --lr_decay=300 --topology=fully_connected start_epoch_phases 0 100 200 --eval_on_average_model=True --steps_eval=400 --lr_warmup_epochs=0 --dataset=cifar100 --seed=1 &
python train_cifar.py --expt_name=solo_B_increase --n_nodes 1 2 4 --lr=0.2 --lr_decay=300 --topology=fully_connected start_epoch_phases 0 100 200 --eval_on_average_model=True --steps_eval=400 --lr_warmup_epochs=0 --dataset=cifar100 --seed=2 &


wait
