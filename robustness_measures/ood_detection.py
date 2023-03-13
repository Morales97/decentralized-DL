'''
Adapted from https://github.com/hendrycks/pre-training/blob/83f5787dea1532a66fd79ef6bbfb8b88e9af9514/uncertainty/CIFAR/test.py
'''
import torch
import numpy as np
import torch.nn.functional as F

import os
import sys
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from helpers.parser import parse_args
from avg_index.search_avg import get_avg_model
from model.model import get_model
from loaders.data import get_data, ROOT_CLUSTER
from helpers.evaluate import evaluate_model
import pdb
import sklearn.metrics as sk

RECALL_LEVEL = 0.95

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


def get_measures(_pos, _neg, recall_level=RECALL_LEVEL):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr


def show_performance(pos, neg, method_name='Ours', recall_level=RECALL_LEVEL):
    '''
    :param pos: 1's class, class to detect, outliers, or wrongly predicted
    example scores
    :param neg: 0's class scores
    '''

    auroc, aupr, fpr = get_measures(pos[:], neg[:], recall_level)

    print('\t\t\t' + method_name)
    print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    print('AUROC:\t\t\t{:.2f}'.format(100 * auroc))
    print('AUPR:\t\t\t{:.2f}'.format(100 * aupr))

def get_ood_scores(model, loader, ood_num_examples, in_dist=False, test_bs=100, use_CE=False):
    _score = []
    _right_score = []
    _wrong_score = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // test_bs and in_dist is False:
                break

            data = data.cuda()

            output = model(data)
            # output = output/torch.norm(output, 2, -1, keepdim=True)
            smax = to_np(F.softmax(output, dim=1))

            if use_CE:  # NOTE use cross entropy instead of MSP
                _score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1))))
            else:
                _score.append(-np.max(smax, axis=1))

            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                if args.use_xent:
                    _right_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[right_indices])
                    _wrong_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[wrong_indices])
                else:
                    _right_score.append(-np.max(smax[right_indices], axis=1))
                    _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()

def ood_gaussian_noise(args, model, test_loader, t_star):
    _, test_confidence, test_correct, _, _ = get_model_calibration_results(model, test_loader, in_dist=True, t=t_star)

    ood_num_examples = len(test_loader.dataset) // 5



if __name__ == '__main__':
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, val_loader, test_loader = get_data(args, args.batch_size[0], args.data_fraction, args.val_fraction)

    if args.resume:
        model = get_model(args, device)
        ckpt = torch.load(args.resume)
        model.load_state_dict(ckpt['state_dict'])

    in_score, right_score, wrong_score = get_ood_scores(model, test_loader, ood_num_examples=0, in_dist=True)

    num_right = len(right_score)
    num_wrong = len(wrong_score)
    print('Error Rate {:.2f}'.format(100 * num_wrong / (num_wrong + num_right)))

    print('\n\nError Detection')
    show_performance(wrong_score, right_score, method_name=args.method_name)

# python robustness_measures/ood_detection.py --net=vgg16 --dataset=cifar100 --resume=/mloraw1/danmoral/checkpoints/cifar100/vgg16/SGD_0.06_s0/checkpoint_last.pth.tar
