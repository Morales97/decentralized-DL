#!/bin/sh

# Actual 16 nodes
# python train_cifar.py --expt_name=FC_16 --n_nodes=16 --topology=fully_connected lr=3.2 --seed=1 --eval_on_average_model=True &
# python train_cifar.py --expt_name=FC_16 --n_nodes=16 --topology=fully_connected lr=3.2 --seed=2 --eval_on_average_model=True &
# python train_cifar.py --expt_name=FC_16 --n_nodes=16 --topology=fully_connected lr=3.2 --seed=0 --eval_on_average_model=True &

# solo. Bx16. DATA SPLIT
# python train_cifar.py --expt_name=solo_B16 --data_split=True --batch_size=2048 --n_nodes=1 --topology=solo lr=3.2 --seed=0 &
# python train_cifar.py --expt_name=solo_B16 --data_split=True --batch_size=2048 --n_nodes=1 --topology=solo lr=3.2 --seed=1 &
# python train_cifar.py --expt_name=solo_B16 --data_split=True --batch_size=2048 --n_nodes=1 --topology=solo lr=3.2 --seed=2 &

# solo. Bx16. data slit. NO LR DECAY
python train_cifar.py --expt_name=C2.9b_solo_NOdecay --data_split=True --batch_size=2048 --n_nodes=1 --topology=solo lr=3.2 --lr_decay=300 --seed=0 &
python train_cifar.py --expt_name=C2.9b_solo_NOdecay --data_split=True --batch_size=2048 --n_nodes=1 --topology=solo lr=3.2 --lr_decay=300 --seed=1 &
python train_cifar.py --expt_name=C2.9b_solo_NOdecay --data_split=True --batch_size=2048 --n_nodes=1 --topology=solo lr=3.2 --lr_decay=300 --seed=2 &
wait