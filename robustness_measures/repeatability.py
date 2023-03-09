
import torch
import numpy as np

import os
import sys

sys.path.insert(0, os.path.join(sys.path[0], '..'))
from helpers.parser import parse_args
from avg_index.search_avg import get_avg_model
from model.model import get_model
from loaders.data import get_data, ROOT_CLUSTER
from helpers.evaluate import evaluate_model, eval_ensemble
from helpers.train_dynamics import get_agreement_metrics
import argparse
import pdb

def load_model(args, path, device):
    model = get_model(args, device)
    ckpt = torch.load(path)
    if args.load_ema:
        alpha = ckpt['best_alpha']
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
    # val_logits, val_confidence, val_correct, val_labels = get_net_results(val_loader, in_dist=True)   # NOTE need to split train in train-val

    if args.resume:
        model1 = load_model(args, args.resume, device)
        model2 = load_model(args, args.resume2, device)
        model3 = load_model(args, args.resume3, device)
        models = [model1, model2, model3]
    else:
        model = get_avg_model(args, start=0.5, end=1)
        # TODO

    _, acc, soft_acc, losses, accs, soft_accs = eval_ensemble(models, test_loader, device)
    _, avg_model_acc, _, _ = eval_ensemble(models, test_loader, device, avg_model=True)
    print('\n ~~~ Models accuracy ~~~')
    for i in range(len(accs)):
        print(f'Model {i}:\tAccuracy: {accs[i]:.2f} \tLoss: {losses[i]:.4f} \tSoft accuracy: {soft_accs[i]:.2f}')
    print(f'(Prediction) Ensemble Accuracy: {acc:.2f} \tSoft accuracy: {soft_acc:.2f}')
    print(f'(Weight) Ensemble Accuracy: {avg_model_acc:.2f}')

    pred_disagreement = np.zeros((len(models), len(models)))
    pred_distance = np.zeros((len(models), len(models)))
    pred_js_div = np.zeros((len(models), len(models)))
    corr_corr = np.zeros((len(models), len(models)))
    incorr_corr = np.zeros((len(models), len(models)))
    incorr_incorr_same = np.zeros((len(models), len(models)))
    incorr_incorr_diff = np.zeros((len(models), len(models)))

    for i, model_i in enumerate(models):
        pred_disagreement[i,i] = 0
        pred_distance[i,i] = 0
        corr_corr[i,i] = 0
        incorr_corr[i,i] = 0
        incorr_incorr_same[i,i] = 0
        incorr_incorr_diff[i,i] = 0

        for j, model_j in enumerate(models[i+1:]):
            j = j+i+1
            
            results = get_agreement_metrics(model_i, model_j, test_loader, device)
            pred_distance[i,j] = results['L2']
            pred_js_div[i,j] = results['JS_div']
            pred_disagreement[i,j] = results['disagreement']
            # corr_corr[i,j] = results['correct-correct']
            # incorr_corr[i,j] = results['correct-incorrect']
            # incorr_incorr_same[i,j] = results['incorrect-incorrect-same']
            # incorr_incorr_diff[i,j] = results['incorrect-incorrect-different']

            

    print('\n ~~~ Prediction disagreement ~~~')
    print('Fraction of test samples prediction with a different class')
    print(pred_disagreement)

    print('\n ~~~ Prediction distance ~~~')
    print('Average L2 norm of (prob1 - prob2) in test samples')
    print(pred_distance)

    print('\n ~~~ Prediction JS divergence ~~~')
    print('Average JS divergence of (prob1 - prob2) in test samples')
    print(pred_js_div)

    # print('\nCorrect-Correct')
    # print(corr_corr)
    # print('Incorrect-Correct')
    # print(incorr_corr)
    # print('Incorrect-Incorrect, same prediction')
    # print(incorr_incorr_same)
    # print('Incorrect-Incorrect, different prediction')
    # print(incorr_incorr_diff)


# python robustness_measures/repeatability.py --net=rn18 --dataset=cifar100 --resume=/mloraw1/danmoral/checkpoints/C4.3_lr0.8_cosine/checkpoint_m0_117001.pth.tar --resume2=/mloraw1/danmoral/checkpoints/C4.3_lr0.8_cosine_1/checkpoint_m0_117001.pth.tar --resume3=/mloraw1/danmoral/checkpoints/C4.3_lr0.8_cosine_2/checkpoint_m0_117001.pth.tar
# python robustness_measures/repeatability.py --net=rn18 --dataset=cifar100 --resume=/mloraw1/danmoral/checkpoints/cifar100/rn18/search_0.8_s0/best_student_acc.pth.tar --resume2=/mloraw1/danmoral/checkpoints/cifar100/rn18/search_0.8_s1/best_student_acc.pth.tar --resume3=/mloraw1/danmoral/checkpoints/cifar100/rn18/search_0.8_s2/best_student_acc.pth.tar
# python robustness_measures/repeatability.py --net=rn18 --dataset=cifar100 --resume=/mloraw1/danmoral/checkpoints/cifar100/rn18/search_0.8_s0/best_ema_acc.pth.tar --resume2=/mloraw1/danmoral/checkpoints/cifar100/rn18/search_0.8_s1/best_ema_acc.pth.tar --resume3=/mloraw1/danmoral/checkpoints/cifar100/rn18/search_0.8_s2/best_ema_acc.pth.tar --load_ema
# python robustness_measures/repeatability.py --net=rn18 --dataset=cifar100 --resume=/mloraw1/danmoral/checkpoints/C4.3_lr0.8_cosine/checkpoint_m0_60841.pth.tar --resume2=/mloraw1/danmoral/checkpoints/C4.3_lr0.8_cosine_1/checkpoint_m0_60841.pth.tar --resume3=/mloraw1/danmoral/checkpoints/C4.3_lr0.8_cosine_2/checkpoint_m0_60841.pth.tar --load_ema
# python robustness_measures/repeatability.py --net=rn18 --dataset=cifar100 --resume=/mloraw1/danmoral/checkpoints/C4.3_lr0.8_cosine/checkpoint_m0_88921.pth.tar --resume2=/mloraw1/danmoral/checkpoints/C4.3_lr0.8_cosine_1/checkpoint_m0_88921.pth.tar --resume3=/mloraw1/danmoral/checkpoints/C4.3_lr0.8_cosine_2/checkpoint_m0_88921.pth.tar --load_ema
# python robustness_measures/repeatability.py --net=rn18 --dataset=cifar100 --resume=/mloraw1/danmoral/checkpoints/C4.3_lr0.8_cosine/checkpoint_m0_60841.pth.tar --resume2=/mloraw1/danmoral/checkpoints/C4.3_lr0.8_cosine_1/checkpoint_m0_88921.pth.tar --resume3=/mloraw1/danmoral/checkpoints/C4.3_lr0.8_cosine_2/checkpoint_m0_117001.pth.tar --load_ema
