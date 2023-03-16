
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


def rank_DP(args, model):
    
    train_loader, val_loader, test_loader = get_data(args, args.batch_size[0], args.data_fraction, val_fraction=0.2)

    confidences = []
    for data, labels in train_loader:
        data = data.cuda()
        labels = labels.cuda()
        probs = F.softmax(model(data), dim=1)
        max_probs = torch.max(probs, dim=1)[0]
        confidences += list(max_probs.tolist())

        if len(confidences) >= 10000:
            confidences = confidences[:10000]
            break

    pdb.set_trace()
    for data, labels in test_loader:
        data = data.cuda()
        labels = labels.cuda()
        probs = F.softmax(model(data), dim=1)
        max_probs = torch.max(probs, dim=1)[0]
        confidences += list(max_probs.tolist())
    pdb.set_trace()

if __name__ == '__main__':
    ''' For debugging purposes '''
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(args, device)
    ckpt = torch.load(args.resume)
    model.load_state_dict(ckpt['state_dict'])
    # model.load_state_dict(ckpt['ema_state_dict_0.996'])
    rank_DP(args, model)

# python robustness_measures/diff_privacy.py --net=rn18 --dataset=cifar100 --resume=/mloraw1/danmoral/checkpoints/cifar100/rn18/val_0.8_s0/checkpoint_last.pth.tar