#!/bin/sh
python train_cifar.py --expt_name=solo_baseline_10k --data_fraction=$1 --n_nodes=1 --lr=0.2 --topology=solo --steps_eval=60 --lr_warmup_epochs=0 --seed=0 &
python train_cifar.py --expt_name=solo_baseline_10k --data_fraction=$1 --n_nodes=1 --lr=0.2 --topology=solo --steps_eval=60 --lr_warmup_epochs=0 --seed=1 &
python train_cifar.py --expt_name=FC_16_10k --data_fraction=$1 --topology=fully_connected --eval_on_average_model=True --lr=3.2 --steps_eval=15 --seed=0 &
python train_cifar.py --expt_name=FC_16_10k --data_fraction=$1 --topology=fully_connected --eval_on_average_model=True --lr=3.2 --steps_eval=15 --seed=1 &
python train_cifar.py --expt_name=C2.2_10k --data_fraction=$1 --topology fully_connected fully_connected fully_connected --local_steps 0 0 0 --n_nodes 8 16 32 --start_epoch_phases 0 100 200 --lr_decay 300 --eval_on_average_model=True --steps_eval=15 --seed=0 &
python train_cifar.py --expt_name=C2.2_10k --data_fraction=$1 --topology fully_connected fully_connected fully_connected --local_steps 0 0 0 --n_nodes 8 16 32 --start_epoch_phases 0 100 200 --lr_decay 300 --eval_on_average_model=True --steps_eval=15 --seed=1 &
wait