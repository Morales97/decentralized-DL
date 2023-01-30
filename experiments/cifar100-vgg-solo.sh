#!/bin/sh
# solo. Bx1
# python train_cifar.py --expt_name=VGG16BN_solo_lr0.02 --dataset=cifar100 --net=vgg16bn --n_nodes=1 --topology=solo --lr=0.02 --wd=5e-4 --epochs=200 --lr_decay 100 175 --lr_warmup_epochs=0 --seed=1 &
# python train_cifar.py --expt_name=VGG16BN_solo_lr0.02 --dataset=cifar100 --net=vgg16bn --n_nodes=1 --topology=solo --lr=0.02 --wd=5e-4 --epochs=200 --lr_decay 100 175 --lr_warmup_epochs=0 --seed=2 &

# solo. Bx16. DATA SPLIT
# python train_cifar.py --expt_name=VGG_solo_B16 --dataset=cifar100 --data_split=True --net=vgg16bn --n_nodes=1 --batch_size=2048 --topology=solo --lr=0.32 --wd=5e-4 --epochs=200 --lr_decay 100 175 --seed=1 &
# python train_cifar.py --expt_name=VGG_solo_B16 --dataset=cifar100 --data_split=True --net=vgg16bn --n_nodes=1 --batch_size=2048 --topology=solo --lr=0.32 --wd=5e-4 --epochs=200 --lr_decay 100 175 --seed=2 &

# solo. Bx16. data slit. NO LR DECAY
python train_cifar.py --expt_name=VGG_solo_B16_NOdecay --dataset=cifar100 --data_split=True --net=vgg16bn --n_nodes=1 --batch_size=2048 --topology=solo --lr=0.32 --wd=5e-4 --epochs=200 --lr_decay=200 --seed=1 &
python train_cifar.py --expt_name=VGG_solo_B16_NOdecay --dataset=cifar100 --data_split=True --net=vgg16bn --n_nodes=1 --batch_size=2048 --topology=solo --lr=0.32 --wd=5e-4 --epochs=200 --lr_decay=200 --seed=2 &
wait