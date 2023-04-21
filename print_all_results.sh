#!/bin/sh
python full_evaluation.py --net=rn18 --dataset=cifar100 --lr=0.8 --eval_on_test=False & 
python full_evaluation.py --net=rn18 --dataset=cifar100 --lr=0.8 --eval_on_test=True &
python full_evaluation.py --net=vgg16 --dataset=cifar100 --lr=0.06 --eval_on_test=False &
python full_evaluation.py --net=vgg16 --dataset=cifar100 --lr=0.06 --eval_on_test=True &
python full_evaluation.py --net=rn18 --dataset=cifar10 --lr=0.4 --eval_on_test=False &
python full_evaluation.py --net=rn18 --dataset=cifar10 --lr=0.4 --eval_on_test=True &
wait
