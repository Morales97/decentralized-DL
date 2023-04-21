#!/bin/sh
python full_evaluation.py --net=rn18 --dataset=cifar100 --lr=0.8 --eval_on_test=False & 
python full_evaluation.py --net=rn18 --dataset=cifar100 --lr=0.8 --eval_on_test=True &
wait