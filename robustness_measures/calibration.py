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



def get_measures(confidence, correct):
    rms = calib_err(confidence, correct, p='2')
    mad = calib_err(confidence, correct, p='1')
    sf1 = soft_f1(confidence, correct)

    return rms, mad, sf1


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

    return np.round(rms_mean/len(models)*100, 2), np.round(mad_mean/len(models)*100, 2), np.round(sf1_mean/len(models)*100, 2)


def calibration_error(model, data_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    probs = None

    for data, labels in data_loader:
        data = data.to(device)
        labels = labels.to(device)
        
        out = model(data)
        batch_probs = F.softmax(out, dim=1)
        if probs is None:
            probs = batch_probs
        else:
            probs = torch.cat((probs, batch_probs), dim=0)
    
    import calibration as cal
    pdb.set_trace()
    calibration_error = cal.get_calibration_error(probs, data_loader.dataset.targets)


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

    calibration_error(model, test_loader)
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
    

# python robustness_measures/calibration.py --net=vgg16 --dataset=cifar100 --resume=/mloraw1/danmoral/checkpoints/cifar100/vgg16/SGD_0.06_s0/checkpoint_last.pth.tar
# python robustness_measures/calibration.py --net=vgg16 --dataset=cifar100 --resume=/mloraw1/danmoral/checkpoints/cifar100/vgg16/search_0.06_s0/checkpoint_last.pth.tar
