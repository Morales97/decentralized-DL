'''
Adapted from https://github.com/hendrycks/pre-training/blob/83f5787dea1532a66fd79ef6bbfb8b88e9af9514/uncertainty/CIFAR/test_calibration.py
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

def calib_err(confidence, correct, p='2', beta=100):
    # beta is target bin size
    idxs = np.argsort(confidence)

    confidence = confidence[idxs]
    correct = correct[idxs]
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0
    total_examples = len(confidence)
    for i in range(len(bins) - 1):
        bin_confidence = confidence[bins[i][0]:bins[i][1]]
        bin_correct = correct[bins[i][0]:bins[i][1]]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))

            if p == '2':
                cerr += num_examples_in_bin / total_examples * np.square(difference)
            elif p == '1':
                cerr += num_examples_in_bin / total_examples * difference
            elif p == 'infty' or p == 'infinity' or p == 'max':
                cerr = np.maximum(cerr, difference)
            else:
                assert False, "p must be '1', '2', or 'infty'"

    if p == '2':
        cerr = np.sqrt(cerr)

    return cerr


def soft_f1(confidence, correct):
    wrong = 1 - correct
    return 2 * ((1 - confidence) * wrong).sum()/(1 - confidence + wrong).sum()


def tune_temp(logits, labels, binary_search=True, lower=0.2, upper=5.0, eps=0.0001):
    logits = np.array(logits)

    if binary_search:
        import torch
        import torch.nn.functional as F

        logits = torch.FloatTensor(logits)
        labels = torch.LongTensor(labels)
        t_guess = torch.FloatTensor([0.5*(lower + upper)]).requires_grad_()

        while upper - lower > eps:
            if torch.autograd.grad(F.cross_entropy(logits / t_guess, labels), t_guess)[0] > 0:
                upper = 0.5 * (lower + upper)
            else:
                lower = 0.5 * (lower + upper)
            t_guess = t_guess * 0 + 0.5 * (lower + upper)

        t = min([lower, 0.5 * (lower + upper), upper], key=lambda x: float(F.cross_entropy(logits / x, labels)))
    else:
        import cvxpy as cx

        set_size = np.array(logits).shape[0]

        t = cx.Variable()

        expr = sum((cx.Minimize(cx.log_sum_exp(logits[i, :] * t) - logits[i, labels[i]] * t)
                    for i in range(set_size)))
        p = cx.Problem(expr, [lower <= t, t <= upper])

        p.solve()   # p.solve(solver=cx.SCS)
        t = 1 / t.value

    return t

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

def get_measures(confidence, correct):
    rms = calib_err(confidence, correct, p='2')
    mad = calib_err(confidence, correct, p='1')
    sf1 = soft_f1(confidence, correct)

    return rms, mad, sf1

def get_measures_roc(_pos, _neg, recall_level=RECALL_LEVEL):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr

to_np = lambda x: x.data.to('cpu').numpy()
concat = lambda x: np.concatenate(x, axis=0)

def get_model_calibration_results(model, data_loader, in_dist=True, t=1, ood_num_examples=0):
    logits = []
    confidence = []
    correct = []
    labels = []
    running_loss = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):    
            if batch_idx >= ood_num_examples // 100 and in_dist is False:   # NOTE 100 is test batch size
                break
            data, target = data.cuda(), target.cuda()

            output = model(data)
            logits.extend(to_np(output).squeeze())

            # if args.use_01:
            #     confidence.extend(to_np(
            #         (F.softmax(output/t, dim=1).max(1)[0] - 1./num_classes)/(1 - 1./num_classes)
            #     ).squeeze().tolist())
            # else:

            confidence.extend(to_np(F.softmax(output/t, dim=1).max(1)[0]).squeeze().tolist())

            if in_dist:
                pred = output.data.max(1)[1]
                correct.extend(pred.eq(target).to('cpu').numpy().squeeze().tolist())
                labels.extend(target.to('cpu').numpy().squeeze().tolist())
            
                loss = F.cross_entropy(output, target, reduction='sum')
                running_loss += loss.cpu().data.numpy()
    
    if not in_dist:
        return logits[:ood_num_examples].copy(), confidence[:ood_num_examples].copy()
    else:
        loss = running_loss / len(data_loader.dataset)
        return logits.copy(), confidence.copy(), correct.copy(), labels.copy(), loss


def eval_calibration(args, models, test_loader):
    print('Ignoring temperature for calibration')
    t_star = 1

    rms_mean, mad_mean, sf1_mean = 0, 0, 0
    for model in models:
        test_logits, test_confidence, test_correct, _, test_loss = get_model_calibration_results(model, test_loader, in_dist=True, t=t_star)
        rms, mad, sf1 = get_measures(np.array(test_confidence), np.array(test_correct))
        rms_mean += rms
        mad_mean += mad
        sf1_mean += sf1

    return np.round(rms_mean/len(models), 2), np.round(mad_mean/len(models), 2), np.round(sf1_mean/len(models), 2)

def print_measures(auroc, aupr, fpr, method_name='Ours', recall_level=RECALL_LEVEL):
    print('\t\t\t\t' + method_name)
    print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    print('AUROC: \t\t\t{:.2f}'.format(100 * auroc))
    print('AUPR:  \t\t\t{:.2f}'.format(100 * aupr))

def get_and_print_results(model, ood_loader, test_confidence, test_correct, ood_num_examples, num_to_avg=1, t_star=1):

    rmss, mads, sf1s = [], [], []
    for _ in range(num_to_avg):
        out_logits, out_confidence = get_model_calibration_results(model, ood_loader, t=t_star, in_dist=False, ood_num_examples=ood_num_examples)
        pdb.set_trace()
        measures = get_measures_roc(
            concat([out_confidence, test_confidence]),
            concat([np.zeros(len(out_confidence)), test_correct]))

        rmss.append(measures[0]); mads.append(measures[1]); sf1s.append(measures[2])

    rms = np.mean(rmss); mad = np.mean(mads); sf1 = np.mean(sf1s)
    # rms_list.append(rms); mad_list.append(mad); sf1_list.append(sf1)

    print_measures(rms, mad, sf1, '')
    return rms, mad, sf1


# /////////////// Gaussian Noise ///////////////

def ood_gaussian_noise(args, model, test_loader, t_star):
    _, test_confidence, test_correct, _, _ = get_model_calibration_results(model, test_loader, in_dist=True, t=t_star)

    ood_num_examples = len(test_loader.dataset) // 5
    # expected_ap = ood_num_examples / (ood_num_examples + len(test_loader.dataset))

    dummy_targets = torch.ones(ood_num_examples)
    ood_data = torch.from_numpy(np.float32(np.clip(
        np.random.normal(size=(ood_num_examples, 3, 32, 32), scale=0.5), -1, 1)))
    ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=True)

    print('\n\nGaussian Noise (sigma = 0.5) Calibration')
    return get_and_print_results(model, ood_loader, test_confidence, test_correct, ood_num_examples)



def eval_ood(args, models, test_loader):
    t_star = 1

    rms_mean, mad_mean, sf1_mean = 0, 0, 0
    for model in models:
        rms, mad, sf1 = ood_gaussian_noise(args, model, test_loader, t_star)
        rms_mean += rms
        mad_mean += mad
        sf1_mean += sf1

    return np.round(rms_mean/len(models), 2), np.round(mad_mean/len(models), 2), np.round(sf1_mean/len(models), 2)


if __name__ == '__main__':
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader, test_loader = get_data(args, args.batch_size[0], args.data_fraction, args.val_fraction)
    # val_logits, val_confidence, val_correct, val_labels = get_net_results(val_loader, in_dist=True)   # NOTE need to split train in train-val

    if args.resume:
        model = get_model(args, device)
        ckpt = torch.load(args.resume)
        model.load_state_dict(ckpt['state_dict'])
        # model.load_state_dict(ckpt['ema_state_dict'])
    else:
        model = get_avg_model(args, start=0.5, end=1)

    # loss, acc = evaluate_model(model, test_loader, device)
    # print(f'Loss: {loss}, Acc: {acc}')
    
    # val_logits, val_confidence, val_correct, val_labels, _ = get_model_calibration_results(model, val_loader, in_dist=True)   # NOTE need to split train in train-val
    # print('\nTuning Softmax Temperature')
    # t_star = tune_temp(val_logits, val_labels)
    # print('Softmax Temperature Tuned. Temperature is {:.3f}'.format(t_star))
    print('Ignoring temperature')
    t_star = 1

    test_logits, test_confidence, test_correct, _, test_loss = get_model_calibration_results(model, test_loader, in_dist=True, t=t_star)
    rms, mad, sf1 = get_measures(np.array(test_confidence), np.array(test_correct))
    print(f'Test Accuracy: {np.sum(test_correct)/10000*100} \t\t Test Loss: {test_loss}')
    print('RMS Calib Error (%): \t\t{:.2f}'.format(100 * rms))
    print('MAD Calib Error (%): \t\t{:.2f}'.format(100 * mad))
    print('Soft F1 Score (%):   \t\t{:.2f}'.format(100 * sf1))
    
    ood_gaussian_noise(args, model, test_loader, t_star)

# python robustness_measures/calibration.py --net=vgg16 --dataset=cifar100 --resume=/mloraw1/danmoral/checkpoints/cifar100/vgg16/SGD_0.06_s0/checkpoint_last.pth.tar
# python robustness_measures/calibration.py --net=vgg16 --dataset=cifar100 --resume=/mloraw1/danmoral/checkpoints/cifar100/vgg16/search_0.06_s0/checkpoint_last.pth.tar
