
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

def evaluate_pgd_attack(model, test_loader, adv=True, epsilon=8./255):
    adversary = attacks.PGD_linf(epsilon=epsilon, num_steps=20, step_size=2./255).cuda()

    model.eval()
    if adv is False:
        torch.set_grad_enabled(False)
    running_loss = 0
    running_acc = 0
    count = 0
    for i, batch in enumerate(test_loader):
        bx = batch[0].cuda()
        by = batch[1].cuda()

        count += by.size(0)

        adv_bx = adversary(model, bx, by) if adv else bx
        with torch.no_grad():
            logits = model(adv_bx) # TODO change this
            # logits = model(adv_bx * 2 - 1) # TODO change this

        loss = F.cross_entropy(logits.data, by, reduction='sum')
        running_loss += loss.cpu().data.numpy()
        running_acc += (torch.max(logits, dim=1)[1] == by).float().sum(0).cpu().data.numpy()
    running_loss /= count
    running_acc /= count

    loss = running_loss
    acc = running_acc

    if adv is False:
        torch.set_grad_enabled(True)
    return loss, acc

def pgd_attack(fmodel, test_loader, epsilon):
    attack = fb.attacks.LinfPGD(steps=20)

    robust_accuracy = 0
    for images, labels in iter(test_loader):
        images = images.cuda()
        images /= 255
        labels = labels.cuda().long()

        _, _, is_adv = attack(fmodel, images, labels, epsilons=epsilon)
        robust_accuracy += 1 - is_adv.float().mean(axis=-1)

    return robust_accuracy / len(test_loader)

def evaluate_adversarial(models, test_loader, epsilon):
    acc_mean = []
    for model in models:
        loss, acc = evaluate_pgd_attack(model, test_loader, epsilon=epsilon)
        acc_mean.append(acc)

    return np.round(np.mean(acc_mean)*100, 2)

if __name__ == '__main__':
    args = parse_args()

    # train_loader, _, test_loader = get_data(args, args.batch_size[0], args.data_fraction)

    test_loader = get_unprocessed_test(args)

    # dataset_loader = datasets.CIFAR100
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    # # transform = transforms.Compose([transforms.ToTensor()])
    # dataset = dataset_loader(
    #     root=ROOT_CLUSTER,
    #     train=False,
    #     transform=transform,
    #     download=True,
    # )
    # test_loader = data.DataLoader(dataset, batch_size=200, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.resume:
        model = get_model(args, device)
        ckpt = torch.load(args.resume)
        #model.load_state_dict(ckpt['state_dict'])
        model.load_state_dict(ckpt['ema_state_dict_0.996'])
    else:
        model = get_avg_model(args, start=0.5, end=1)

    if args.dataset == 'cifar100':
        preprocessing = dict(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761], axis=-3)

    fmodel = fb.PyTorchModel(model, bounds=(0,1), preprocessing=preprocessing) 
    images = torch.Tensor(test_loader.dataset.data).cuda()
    images = torch.transpose(images, 1, 3) # convert to NCHW
    images = torch.transpose(images, 2, 3) 
    labels = torch.Tensor(test_loader.dataset.targets).cuda().long()

    images = images[:1000]  # to fit in memory
    labels = labels[:1000]

    images /= 255
    clean_acc = fb.utils.accuracy(fmodel, images, labels)
    print(f'Clean accuracy: {clean_acc*100}')

    attack = fb.attacks.LinfPGD(steps=20)
    epsilon = 8/255
    _, _, is_adv = attack(fmodel, images, labels, epsilons=epsilon)
    robust_accuracy = 1 - is_adv.float().mean(axis=-1)
    print(f'Robust accuracy for epsilon {epsilon}: {robust_accuracy*100}')

    epsilon = 2/255
    _, _, is_adv = attack(fmodel, images, labels, epsilons=epsilon)
    robust_accuracy = 1 - is_adv.float().mean(axis=-1)
    print(f'Robust accuracy for epsilon {epsilon}: {robust_accuracy*100}')

    robust_accuracy = pgd_attack(fmodel, test_loader, epsilon)
    print(f'Robust accuracy for epsilon {epsilon}: {robust_accuracy*100}')
    
    # epsilon = 8./255
    # loss, acc = evaluate_pgd_attack(model, test_loader, epsilon=epsilon)
    # print(f'Adversarial Test Accuracy (eps={epsilon}): {acc} \t Advesarial Test Loss: {loss}')

    # epsilon = 4./255
    # loss, acc = evaluate_pgd_attack(model, test_loader, epsilon=epsilon)
    # print(f'Adversarial Test Accuracy (eps={epsilon}): {acc} \t Advesarial Test Loss: {loss}')
    
    # epsilon = 2./255
    # loss, acc = evaluate_pgd_attack(model, test_loader, epsilon=epsilon)
    # print(f'Adversarial Test Accuracy (eps={epsilon}): {acc} \t Advesarial Test Loss: {loss}')

    # loss, acc = evaluate_pgd_attack(model, test_loader, adv=False)
    # print(f'Test Accuracy: {acc} \t Test Loss: {loss}')

# python adversarial/adv_eval.py --net=rn18 --dataset=cifar100 --expt_name=C4.3_lr0.8_cosine
# python adversarial/adv_eval.py --net=rn18 --dataset=cifar100 --resume=/mloraw1/danmoral/checkpoints/cifar100/rn18/EMA_acc_1.2_s0/checkpoint_last.pth.tar