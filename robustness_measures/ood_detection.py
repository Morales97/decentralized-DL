'''
Adapted from https://github.com/hendrycks/pre-training/blob/83f5787dea1532a66fd79ef6bbfb8b88e9af9514/uncertainty/CIFAR/test.py
'''
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

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
from skimage.filters import gaussian as gblur

RECALL_LEVEL = 0.95

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=RECALL_LEVEL, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])

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


def show_performance(pos, neg, recall_level=RECALL_LEVEL):
    '''
    :param pos: 1's class, class to detect, outliers, or wrongly predicted
    example scores
    :param neg: 0's class scores
    '''

    auroc, aupr, fpr = get_measures(pos[:], neg[:], recall_level)

    print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    print('AUROC:\t\t\t{:.2f}'.format(100 * auroc))
    print('AUPR:\t\t\t{:.2f}'.format(100 * aupr))

def get_ood_scores(model, loader, ood_num_examples, in_dist=False, test_bs=100, use_CE=False):   # NOTE still not sure about the difference between CE and MSP
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

                if use_CE:
                    _right_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[right_indices])
                    _wrong_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[wrong_indices])
                else:
                    _right_score.append(-np.max(smax[right_indices], axis=1))
                    _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()

def print_measures(auroc, aupr, fpr, recall_level=RECALL_LEVEL):
    print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    print('AUROC: \t\t\t{:.2f}'.format(100 * auroc))
    print('AUPR:  \t\t\t{:.2f}'.format(100 * aupr))

def get_and_print_results(model, ood_loader, ood_num_examples, in_score, num_to_avg=1):

    aurocs, auprs, fprs = [], [], []
    for _ in range(num_to_avg):
        out_score = get_ood_scores(model, ood_loader, ood_num_examples)
        measures = get_measures(out_score, in_score)
        aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])

    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    # auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)

    print_measures(auroc, aupr, fpr)
    return auroc, aupr, fpr

def ood_gaussian_noise(args, model, ood_num_examples, in_score):

    dummy_targets = torch.ones(ood_num_examples)
    ood_data = torch.from_numpy(np.float32(np.clip(
        np.random.normal(size=(ood_num_examples, 3, 32, 32), scale=0.5), -1, 1)))
    ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=True)

    print('\nGaussian Noise (sigma = 0.5)')
    return get_and_print_results(model, ood_loader, ood_num_examples, in_score)

def ood_rademacher_noise(args, model, ood_num_examples, in_score):
    dummy_targets = torch.ones(ood_num_examples)
    ood_data = torch.from_numpy(np.random.binomial(
        n=1, p=0.5, size=(ood_num_examples, 3, 32, 32)).astype(np.float32)) * 2 - 1
    ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=True)

    print('\nRademacher Noise Detection')
    return get_and_print_results(model, ood_loader, ood_num_examples, in_score)

def ood_random_images(args, model, ood_num_examples, in_score):
    ood_loader = get_ood_loader()
    return get_and_print_results(model, ood_loader, ood_num_examples, in_score)

def ood_blob(args, model, ood_num_examples, in_score):

    ood_data = np.float32(np.random.binomial(n=1, p=0.7, size=(ood_num_examples, 32, 32, 3)))
    for i in range(ood_num_examples):
        ood_data[i] = gblur(ood_data[i], sigma=1.5, multichannel=False)
        ood_data[i][ood_data[i] < 0.75] = 0.0

    dummy_targets = torch.ones(ood_num_examples)
    ood_data = torch.from_numpy(ood_data.transpose((0, 3, 1, 2))) * 2 - 1
    ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=True,
                                            num_workers=args.prefetch, pin_memory=True)

    print('\nBlob Detection')
    return get_and_print_results(model, ood_loader, len(ood_loader.dataset), in_score)

def compute_average_ood(args, models, ood_num_examples, in_scores, ood_fn):
    '''
    Average OOD across models for the specified `ood_fn` function
    '''
    auroc_mean, aupr_mean, fpr_mean = 0, 0, 0
    for i, model in enumerate(models):
        auroc, aupr, fpr = ood_fn(args, model, ood_num_examples, in_scores[i])
        auroc_mean += auroc
        aupr_mean += aupr
        fpr_mean += fpr
    return np.round(auroc_mean/len(models)*100, 2), np.round(aupr_mean/len(models)*100, 2), np.round(fpr_mean/len(models)*100, 2)

def eval_ood(args, models, test_loader):
    ood_num_examples = len(test_loader.dataset) // 5
    auroc_list, aupr_list, fpr_list = [], [], []
    in_scores = []
    for model in models:
        in_score, right_score, wrong_score = get_ood_scores(model, test_loader, ood_num_examples, in_dist=True)
        in_scores.append(in_score)
    
    auroc, aupr, fpr = compute_average_ood(args, models, ood_num_examples, in_scores, ood_gaussian_noise)
    print(f'OOD Detection - Gaussian noise. AUROC: {auroc} \t AUPR {aupr} \t FPR {fpr}')
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)
    
    auroc, aupr, fpr = compute_average_ood(args, models, ood_num_examples, in_scores, ood_rademacher_noise)
    print(f'OOD Detection - Rademacher noise. AUROC: {auroc} \t AUPR {aupr} \t FPR {fpr}')
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)

    # auroc, aupr, fpr = compute_average_ood(args, models, ood_num_examples, in_scores, ood_blob)
    # print(f'OOD Detection - Blop. AUROC: {auroc} \t AUPR {aupr} \t FPR {fpr}')
    # auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)

    auroc, aupr, fpr = compute_average_ood(args, models, ood_num_examples, in_scores, ood_random_images)
    print(f'OOD Detection 300K random images. AUROC: {auroc} \t AUPR {aupr} \t FPR {fpr}')
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)

    return np.array(auroc_list).mean(), np.array(aupr_list).mean(), np.array(fpr_list).mean() 

def eval_ood_random_images(args, models, test_loader):
    auroc_list, aupr_list, fpr_list = [], [], []
    in_scores = []
    for model in models:
        in_score, right_score, wrong_score = get_ood_scores(model, test_loader, 0, in_dist=True)
        in_scores.append(in_score)
    
    auroc, aupr, fpr = compute_average_ood(args, models, len(test_loader.dataset) // 5, in_scores, ood_random_images)
    print(f'OOD Detection 300K random images. AUROC: {auroc} \t AUPR {aupr} \t FPR {fpr}')
    return auroc, aupr, fpr

def get_ood_loader():
    ood_dataset = np.load(ROOT_CLUSTER + str('/OOD_detection/300K_random_images.npy'))
    normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    transform = transforms.Compose([transforms.ToTensor(), normalize]),
    ood_data = CustomDataset(ood_dataset, transform=transforms.Compose([transforms.ToTensor(), normalize]))
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=False)
    return ood_loader

class CustomDataset(Dataset):
    def __init__(self, dataset, transform=None):
        super(CustomDataset, self).__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        if self.transform:
            data = self.transform(data)

        return data, 0 # 0 is the class


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
    show_performance(wrong_score, right_score)

# python robustness_measures/ood_detection.py --net=vgg16 --dataset=cifar100 --resume=/mloraw1/danmoral/checkpoints/cifar100/vgg16/SGD_0.06_s0/checkpoint_last.pth.tar
