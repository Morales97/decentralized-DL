
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import pdb

sys.path.insert(0, os.path.join(sys.path[0], '..'))
from helpers.parser import parse_args
from loaders.data import get_data
from model.model import get_model

# TODO implement membership inference attack of Yeom et al
# run a fwd pass on final model on train set -> get expected train loss per sample
# define a 20k set (10k random train + 10k test) to evaluate DP on
# can also introduce STD of train loss and use it to refine DP prediction
# more elaborate membership attacks: Shkori et al (multiple shadow models) and Salem et al (one shadow model)
# Lastly, Tramer et al (2022) propose a superior membership attack


def rank_DP(args, model, train_loader, test_loader, use_labels=True):
    '''
    use_labels = False -> use top confidence (no label knowledge)
    use_labels = True -> use confidence on the correct label (label knowledge)
    '''

    confidences = []
    for data, labels in train_loader:
        data = data.cuda()
        labels = labels.cuda()
        probs = F.softmax(model(data), dim=1)
        if not use_labels:
            conf = torch.max(probs, dim=1)[0]
        else:
            conf = probs.gather(1, labels.unsqueeze(1)).squeeze()
        confidences += list(conf.tolist())

        if len(confidences) >= 10000:
            confidences = confidences[:10000]
            break

    assert len(test_loader.dataset) == 10000
    for data, labels in test_loader:
        data = data.cuda()
        labels = labels.cuda()
        probs = F.softmax(model(data), dim=1)
        if not use_labels:
            conf = torch.max(probs, dim=1)[0]
        else:
            conf = probs.gather(1, labels.unsqueeze(1)).squeeze()
        confidences += list(conf.tolist())

    sorted_idxs = np.argsort(confidences)   # Sorted from lowest to highest confidence
    test_idxs = sorted_idxs >= 10000    
    membership_attack_acc = (1 - test_idxs[:10000].sum() / 20000) * 100   # percentage of train samples in top 50% confidence
    return membership_attack_acc

def eval_DP_ranking(args, models):
    train_loader, val_loader, test_loader = get_data(args, args.batch_size[0], args.data_fraction, val_fraction=0.2)

    dp_acc_mean = 0
    for model in models:
        dp_acc = rank_DP(args, model, train_loader, test_loader)
        dp_acc_mean += dp_acc

    return np.round(dp_acc_mean/len(models), 2)

if __name__ == '__main__':
    ''' For debugging purposes '''
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(args, device)
    ckpt = torch.load(args.resume)
    if 'ema' in args.resume:
        alpha = ckpt(['best_alpha'])
        model.load_state_dict(ckpt['ema_state_dict_'+str(alpha)])
    else:
        model.load_state_dict(ckpt['state_dict'])
    
    train_loader, val_loader, test_loader = get_data(args, args.batch_size[0], args.data_fraction, val_fraction=0.2)
    dp_acc = rank_DP(args, model, train_loader, test_loader)
    print(f'Ranking Membership Attack Accuracy: {dp_acc:.2f}')

# python robustness_measures/diff_privacy.py --net=rn18 --dataset=cifar100 --resume=/mloraw1/danmoral/checkpoints/cifar100/rn18/val_0.8_s0/checkpoint_last.pth.tar