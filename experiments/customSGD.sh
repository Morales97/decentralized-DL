#!/bin/sh

python train_cifar_customSGD.py --expt_name=a$1_b$2 --project=MLO-optimizer --opt=customSGD --custom_a=$1 --custom_b=$2 --lr=$3 --n_nodes=1 --topology=solo --epochs=50 --lr_decay=100 --lr_warmup_epochs=0 --data_split=True --steps_eval=400 --net=rn18 &
python train_cifar_customSGD.py --expt_name=a$4_b$5 --project=MLO-optimizer --opt=customSGD --custom_a=$4 --custom_b=$5 --lr=$6 --n_nodes=1 --topology=solo --epochs=50 --lr_decay=100 --lr_warmup_epochs=0 --data_split=True --steps_eval=400 --net=rn18 &
python train_cifar_customSGD.py --expt_name=a$7_b$8 --project=MLO-optimizer --opt=customSGD --custom_a=$7 --custom_b=$8 --lr=$9 --n_nodes=1 --topology=solo --epochs=50 --lr_decay=100 --lr_warmup_epochs=0 --data_split=True --steps_eval=400 --net=rn18 &

wait