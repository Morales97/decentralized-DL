'''
Adapted from https://github.com/hendrycks/pre-training/blob/83f5787dea1532a66fd79ef6bbfb8b88e9af9514/uncertainty/CIFAR/test_calibration.py
'''

import torch
import numpy as np
import torch.nn.functional as F

import os
import sys
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from robustness_measures.temperature_scaling import ModelWithTemperature
from helpers.parser import parse_args
from avg_index.search_avg import get_avg_model
from model.model import get_model
from loaders.data import get_data, ROOT_CLUSTER
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


@torch.no_grad()
def eval_calibration(args, models, val_loader, test_loader):
    '''get_calibration_error from https://github.com/p-lambda/verified_calibration/blob/master/calibration/utils.py'''
    import calibration as cal
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ece_mean, ece_temp_mean = [], []
    for model in models:
        probs = None
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            
            out = model(data)
            batch_probs = F.softmax(out, dim=1)
            if probs is None:
                probs = batch_probs
            else:
                probs = torch.cat((probs, batch_probs), dim=0)
        
        targets = test_loader.dataset.targets
        # ece_error = cal.get_ece(probs.detach().cpu(), targets)
        ece_error = cal.get_ece_em(probs.detach().cpu(), targets, num_bins=100) # equal-mass binning
        # ece_error = cal.lower_bound_scaling_ce(probs.detach().cpu(), targets, p=2, debias=False, num_bins=15, binning_scheme=cal.get_equal_bins, mode='top-label') # equal-mass binning and L2 cal error
        ece_mean.append(ece_error)

        # calibrate (Temperature scaling, from https://github.com/gpleiss/temperature_scaling)
        scaled_model = ModelWithTemperature(model)
        scaled_model.set_temperature(val_loader)
        probs_scaled = None
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            
            out = scaled_model(data)
            batch_probs_scaled = F.softmax(out, dim=1)
            if probs_scaled is None:
                probs_scaled = batch_probs_scaled
            else:
                probs_scaled = torch.cat((probs_scaled, batch_probs_scaled), dim=0)
        # ece_temperature = cal.get_ece(probs_scaled.detach().cpu(), targets)
        ece_temperature = cal.get_ece_em(probs_scaled.detach().cpu(), targets, num_bins=100)  # equal-mass binning
        # ece_error = cal.lower_bound_scaling_ce(probs_scaled.detach().cpu(), targets, p=2, debias=False, num_bins=15, binning_scheme=cal.get_equal_bins, mode='top-label') # equal-mass binning and L2 cal error

        ece_temp_mean.append(ece_temperature)

    ece_mean = np.array(ece_mean)
    ece_temp_mean = np.array(ece_temp_mean)
    pdb.set_trace()
    return np.round(ece_mean.mean()*100, 2), np.round(ece_mean.std()*100, 2), np.round(ece_temp_mean.mean()*100, 2), np.round(ece_temp_mean.std()*100, 2)

@torch.no_grad()
def calibration_error(model, data_loader, val_loader):
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
    labels = data_loader.dataset.targets
    # conf = probs.gather(1, torch.Tensor(labels).cuda().long().unsqueeze(1)).squeeze()
    calibration_error = cal.get_calibration_error(probs.detach().cpu(), data_loader.dataset.targets)
    # calibration_error = cal.get_ece(probs.detach().cpu(), data_loader.dataset.targets)
    
    # calibrate
    val_probs = None
    for data, labels in val_loader:
        data = data.to(device)
        labels = labels.to(device)
        
        out = model(data)
        batch_probs = F.softmax(out, dim=1)
        if val_probs is None:
            val_probs = batch_probs
        else:
            val_probs = torch.cat((val_probs, batch_probs), dim=0)
    
    
    calibrator = cal.PlattBinnerMarginalCalibrator(10000, num_bins=10)
    # calibrator = cal.PlattCalibrator(10000, num_bins=10)
    np_labels = np.array(val_loader.dataset.dataset.targets)
    val_labels = np_labels[val_loader.dataset.indices]
    pdb.set_trace()
    # val_probs = val_probs.gather(1, torch.Tensor(val_labels).cuda().long().unsqueeze(1)).squeeze()
    calibrator.train_calibration(val_probs.detach().cpu().numpy(), val_labels)
    calibrated_probs = calibrator.calibrate(probs.detach().cpu().numpy())
    new_cal_error = cal.get_calibration_error(calibrated_probs, data_loader.dataset.targets)
    # new_cal_error = cal.get_ece(calibrated_probs, data_loader.dataset.targets)

    pdb.set_trace()


if __name__ == '__main__':
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader, test_loader = get_data(args, args.batch_size[0], args.data_fraction, args.val_fraction)
    # val_logits, val_confidence, val_correct, val_labels = get_net_results(val_loader, in_dist=True)   # NOTE need to split train in train-val

    if args.resume:
        model = get_model(args, device)
        ckpt = torch.load(args.resume)
        model.load_state_dict(ckpt['state_dict'])
        # alpha = ckpt['best_alpha']
        # model.load_state_dict(ckpt[f'ema_state_dict_{alpha}'])
    else:
        model = get_avg_model(args, start=0.5, end=1)

    with torch.no_grad():
        # Compute ECE
        probs = None
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            
            out = model(data)
            batch_probs = F.softmax(out, dim=1)
            if probs is None:
                probs = batch_probs
            else:
                probs = torch.cat((probs, batch_probs), dim=0)
            
        import calibration as cal
        # ece = cal.get_ece(probs.detach().cpu(), test_loader.dataset.targets)
        ece = cal.get_ece_em(probs.detach().cpu(), test_loader.dataset.targets, num_bins=100)
        print(f'ECE: \t{ece}')


        # Apply temperature scaling
        scaled_model = ModelWithTemperature(model)
        scaled_model.set_temperature(val_loader)

        # Compute ECE with temperature scaling
        probs_scaled = None
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            
            out = scaled_model(data)
            batch_probs_scaled = F.softmax(out, dim=1)
            if probs_scaled is None:
                probs_scaled = batch_probs_scaled
            else:
                probs_scaled = torch.cat((probs_scaled, batch_probs_scaled), dim=0)
        
        # ece_temperature = cal.get_ece(probs_scaled.detach().cpu(), test_loader.dataset.targets)
        ece_temperature = cal.get_ece_em(probs_scaled.detach().cpu(), test_loader.dataset.targets, num_bins=100)
        print(f'ECE after temperature scaling: \t{ece_temperature}')

    

# python robustness_measures/model_calibration.py --net=vgg16 --dataset=cifar100 --resume=/mloraw1/danmoral/checkpoints/cifar100/vgg16/val_0.06_s0/checkpoint_last.pth.tar
# python robustness_measures/model_calibration.py --net=vgg16 --dataset=cifar100 --resume=/mloraw1/danmoral/checkpoints/cifar100/vgg16/val_0.06_s0/best_ema_loss.pth.tar
# python robustness_measures/model_calibration.py --net=vgg16 --dataset=cifar100 --resume=/mloraw1/danmoral/checkpoints/cifar100/vgg16/search_0.06_s0/checkpoint_last.pth.tar

# python robustness_measures/model_calibration.py --net=rn18 --dataset=cifar100 --resume=/mloraw1/danmoral/checkpoints/cifar100/rn18/val_0.8_s0/checkpoint_last.pth.tar
# python robustness_measures/model_calibration.py --net=widern28 --dataset=cifar100 --resume=/mloraw1/danmoral/checkpoints/cifar100/widern28/val_0.1_s0/checkpoint_last.pth.tar
# python robustness_measures/model_calibration.py --net=widern28 --dataset=cifar100 --resume=/mloraw1/danmoral/checkpoints/cifar100/widern28/val_0.1_s0/best_ema_loss.pth.tar
