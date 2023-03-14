
import torch
import numpy as np

import os
import sys


sys.path.insert(0, os.path.join(sys.path[0], '..'))
from helpers.parser import parse_args
from model.model import get_model
from loaders.data import get_data, ROOT_CLUSTER
from helpers.evaluate import eval_on_cifar_corrputed_test, evaluate_model, eval_ensemble
from helpers.train_dynamics import get_agreement_metrics
import argparse
import pdb


def load_model(args, path, device):
    model = get_model(args, device)
    ckpt = torch.load(path)
    if args.load_ema:
        alpha = ckpt['best_alpha']
        print(f'Loading EMA with alpha={alpha} from epoch={ckpt["epoch"]}')
        model.load_state_dict(ckpt['ema_state_dict_' + str(alpha)])
    else:
        model.load_state_dict(ckpt['state_dict'])

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--resume2', type=str, help='Second model to compare')
    parser.add_argument('--resume3', type=str, help='Third model to compare')
    parser.add_argument('--load_ema', action='store_true', help='load EMA models, not students')
    args = parse_args(parser)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader, test_loader = get_data(args, args.batch_size[0], args.data_fraction, args.val_fraction)

    if args.resume:
        model1 = load_model(args, args.resume, device)
        model2 = load_model(args, args.resume2, device)
        model3 = load_model(args, args.resume3, device)
        models = [model1, model2, model3]
    else:
        pass
        # TODO

    # _, acc, soft_acc, losses, accs, soft_accs = eval_ensemble(models, test_loader, device)
    # print('\n ~~~ Models accuracy on original Test set ~~~')
    # for i in range(len(accs)):
    #     print(f'Model {i}:\tAccuracy: {accs[i]:.2f} \tLoss: {losses[i]:.4f} \tSoft accuracy: {soft_accs[i]:.2f}')
    # print(f'(Prediction) Ensemble Accuracy: {acc:.2f} \tSoft accuracy: {soft_acc:.2f}')

    # RAND-AUGMENT
    # test_loader = get_data(args, batch_size=100, test_transforms='RandAugment')
    # _, acc, soft_acc, losses, accs, soft_accs = eval_ensemble(models, test_loader, device)
    # print('\n ~~~ Models accuracy on augmented Test set 1~~~')
    # for i in range(len(accs)):
    #     print(f'Model {i}:\tAccuracy: {accs[i]:.2f} \tLoss: {losses[i]:.4f} \tSoft accuracy: {soft_accs[i]:.2f}')
    # print(f'(Prediction) Ensemble Accuracy: {acc:.2f} \tSoft accuracy: {soft_acc:.2f}')

    # _, acc, soft_acc, losses, accs, soft_accs = eval_ensemble(models, test_loader, device)
    # print('\n ~~~ Models accuracy on augmented Test set 2~~~')
    # for i in range(len(accs)):
    #     print(f'Model {i}:\tAccuracy: {accs[i]:.2f} \tLoss: {losses[i]:.4f} \tSoft accuracy: {soft_accs[i]:.2f}')
    # print(f'(Prediction) Ensemble Accuracy: {acc:.2f} \tSoft accuracy: {soft_acc:.2f}')

    # _, acc, soft_acc, losses, accs, soft_accs = eval_ensemble(models, test_loader, device)
    # print('\n ~~~ Models accuracy on augmented Test set 3~~~')
    # for i in range(len(accs)):
    #     print(f'Model {i}:\tAccuracy: {accs[i]:.2f} \tLoss: {losses[i]:.4f} \tSoft accuracy: {soft_accs[i]:.2f}')
    # print(f'(Prediction) Ensemble Accuracy: {acc:.2f} \tSoft accuracy: {soft_acc:.2f}')

    # CORRUPTED CIFAR

    # NOTE DO CHECK OUT ROBUSTBENCH: https://github.com/RobustBench/robustbencht
    # they have a super nice cifar100c loader: load_cifar10c(n_examples=1000, corruptions=corruptions, severity=5)
    acc = eval_on_cifar_corrputed_test(model1, 'cifar100-C', device, root=ROOT_CLUSTER, distortions=['shot_noise'])


    # acc1 = eval_on_cifar_corrputed_test(model1, 'cifar100-C', device, root=ROOT_CLUSTER)
    # acc2 = eval_on_cifar_corrputed_test(model2, 'cifar100-C', device, root=ROOT_CLUSTER)
    # acc3 = eval_on_cifar_corrputed_test(model3, 'cifar100-C', device, root=ROOT_CLUSTER)
    # print(f'Mean robust accuracy for the 3 models: {(acc1+acc2+acc3)/3}')

# python robustness_measures/img_transforms.py --net=rn18 --dataset=cifar100 --resume=/mloraw1/danmoral/checkpoints/cifar100/rn18/search_0.8_s0/best_student_acc.pth.tar --resume2=/mloraw1/danmoral/checkpoints/cifar100/rn18/search_0.8_s1/best_student_acc.pth.tar --resume3=/mloraw1/danmoral/checkpoints/cifar100/rn18/search_0.8_s2/best_student_acc.pth.tar
# python robustness_measures/img_transforms.py --net=rn18 --dataset=cifar100 --resume=/mloraw1/danmoral/checkpoints/cifar100/rn18/search_0.8_s0/best_ema_acc.pth.tar --resume2=/mloraw1/danmoral/checkpoints/cifar100/rn18/search_0.8_s1/best_ema_acc.pth.tar --resume3=/mloraw1/danmoral/checkpoints/cifar100/rn18/search_0.8_s2/best_ema_acc.pth.tar --load_ema
# python robustness_measures/img_transforms.py --net=rn18 --dataset=cifar100 --resume=/mloraw1/danmoral/checkpoints/cifar100/rn18/search_0.8_s0/best_ema_loss.pth.tar --resume2=/mloraw1/danmoral/checkpoints/cifar100/rn18/search_0.8_s1/best_ema_loss.pth.tar --resume3=/mloraw1/danmoral/checkpoints/cifar100/rn18/search_0.8_s2/best_ema_loss.pth.tar --load_ema
