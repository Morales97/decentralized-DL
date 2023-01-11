#!/bin/sh
python train_cifar.py --n_nodes=1 --lr=0.2 --topology=solo --steps_eval=1600 --net=vgg --epochs=100