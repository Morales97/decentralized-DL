#!/bin/sh
#python full_evaluation.py --net=rn18 --dataset=cifar100 --lr=0.8 --eval_on_test=False & 
#python full_evaluation.py --net=rn18 --dataset=cifar100 --lr=0.8 --eval_on_test=True &
#python full_evaluation.py --net=vgg16 --dataset=cifar100 --lr=0.06 --eval_on_test=False &
#python full_evaluation.py --net=vgg16 --dataset=cifar100 --lr=0.06 --eval_on_test=True &
#python full_evaluation.py --net=rn18 --dataset=cifar10 --lr=0.4 --eval_on_test=False &
#python full_evaluation.py --net=rn18 --dataset=cifar10 --lr=0.4 --eval_on_test=True &
#wait
#python full_evaluation.py --net=widern28 --dataset=cifar100 --lr=0.1 --eval_on_test=False &
#python full_evaluation.py --net=widern28 --dataset=cifar100 --lr=0.1 --eval_on_test=True &
#python full_evaluation.py --net=rn18 --dataset=tiny-in --lr=0.8 --eval_on_test=False &
#python full_evaluation.py --net=rn18 --dataset=tiny-in --lr=0.8 --eval_on_test=True &
python full_evaluation.py --net=rn34 --dataset=cifar100 --lr=0.8 --label_noise=40 --eval_on_test=False &
python full_evaluation.py --net=rn34 --dataset=cifar100 --lr=0.8 --label_noise=40 --eval_on_test=True &
python full_evaluation.py --net=rn34 --dataset=cifar10 --lr=0.8 --label_noise=worse_label --eval_on_test=True &
python full_evaluation.py --net=rn34 --dataset=cifar10 --lr=0.8 --label_noise=worse_label --eval_on_test=False &
wait