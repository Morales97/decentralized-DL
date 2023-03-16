
import torch
import torch.nn.functional as F
import numpy as np

import os
import sys
import foolbox as fb
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from adversarial import attacks
from helpers.parser import parse_args
from loaders.data import get_data, ROOT_CLUSTER, get_unprocessed_test
from model.model import get_model
import pdb

def pgd_attack(fmodel, test_loader, epsilon):
    attack = fb.attacks.LinfPGD(steps=20)

    robust_accuracy = 0
    for data, target in iter(test_loader):
        data = data.cuda()
        # data = torch.transpose(data, 2, 3) 
        target = target.cuda().long()

        _, _, is_adv = attack(fmodel, data, target, epsilons=epsilon)
        robust_accuracy += 1 - is_adv.float().mean(axis=-1)

    return robust_accuracy / len(test_loader)

def evaluate_adversarial(args, models, epsilon):
    acc_mean = []

    if args.dataset == 'cifar100':
        preprocessing = dict(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761], axis=-3)

    if args.dataset == 'tiny-in':
        preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    
    test_loader = get_unprocessed_test(args)

    for model in models:
        fmodel = fb.PyTorchModel(model, bounds=(0,1), preprocessing=preprocessing) 
        robust_acc = pgd_attack(fmodel, test_loader, epsilon)
        acc_mean.append(robust_acc.cpu())

    return np.round(np.mean(acc_mean)*100, 2)

if __name__ == '__main__':
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_loader = get_unprocessed_test(args)

    if args.resume:
        model = get_model(args, device)
        ckpt = torch.load(args.resume)
        #model.load_state_dict(ckpt['state_dict'])
        model.load_state_dict(ckpt['ema_state_dict_0.996'])
    else:
        model = get_avg_model(args, start=0.5, end=1)

    if args.dataset == 'cifar100':
        preprocessing = dict(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761], axis=-3)

    # init foolbox model and data
    fmodel = fb.PyTorchModel(model, bounds=(0,1), preprocessing=preprocessing) 
    images = torch.Tensor(test_loader.dataset.data).cuda()
    images = torch.transpose(images, 1, 3) # convert to NCHW
    images = torch.transpose(images, 2, 3) 
    images /= 255
    labels = torch.Tensor(test_loader.dataset.targets).cuda().long()

    images = images[:1000]  # to fit in memory
    labels = labels[:1000]

    clean_acc = fb.utils.accuracy(fmodel, images, labels)
    print(f'Clean accuracy: {clean_acc*100}')

    attack = fb.attacks.LinfPGD(steps=20)
    epsilon = 8/255
    _, _, is_adv = attack(fmodel, images, labels, epsilons=epsilon)
    robust_accuracy = 1 - is_adv.float().mean(axis=-1)
    print(f'Robust accuracy for epsilon {epsilon}: {robust_accuracy*100}')

    # epsilon = 2/255
    # _, _, is_adv = attack(fmodel, images, labels, epsilons=epsilon)
    # robust_accuracy = 1 - is_adv.float().mean(axis=-1)
    # print(f'Robust accuracy for epsilon {epsilon}: {robust_accuracy*100}')

    robust_accuracy = pgd_attack(fmodel, test_loader, epsilon)
    print(f'Robust accuracy for epsilon {epsilon}: {robust_accuracy*100}')

# python adversarial/adv_eval.py --net=rn18 --dataset=cifar100 --expt_name=C4.3_lr0.8_cosine
# python adversarial/adv_eval.py --net=rn18 --dataset=cifar100 --resume=/mloraw1/danmoral/checkpoints/cifar100/rn18/EMA_acc_1.2_s0/checkpoint_last.pth.tar