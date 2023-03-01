
import torch
import numpy as np

import os
import sys
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from helpers.parser import parse_args
from avg_index.search_avg import get_avg_model
from model.model import get_model
from loaders.data import get_data, ROOT_CLUSTER
from helpers.evaluate import evaluate_model
from helpers.train_dynamics import get_prediction_disagreement
import argparse
import pdb

def load_model(args, path, device):
    model = get_model(args, device)
    ckpt = torch.load(path)
    if args.load_ema:
        model.load_state_dict(ckpt['ema_state_dict'])
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

    train_loader, test_loader = get_data(args, args.batch_size[0], args.data_fraction)
    # val_logits, val_confidence, val_correct, val_labels = get_net_results(val_loader, in_dist=True)   # NOTE need to split train in train-val

    if args.resume:
        model1 = load_model(args, args.resume, device)
        model2 = load_model(args, args.resume2, device)
        model3 = load_model(args, args.resume3, device)
        models = [model1, model2, model3]
    else:
        model = get_avg_model(args, start=0.5, end=1)
        # TODO


    pred_disagreement = np.zeros((len(models), len(models)))
    pred_distance = np.zeros((len(models), len(models)))

    for i, model_i in enumerate(models)):
        pred_disagreement[i,i] = 0
        pred_distance[i,i] = 0

        for j, model_j in enumerate(models[i+1:])):
            pred_distance[i,j], pred_disagreement[i,j] = get_prediction_disagreement(model_i, model_j, test_loader, device)
            pred_distance[j,i], pred_disagreement[j,i] = pred_distance[i,j], pred_disagreement[i,j]

    print('\n ~~~ Prediction disagreement ~~~')
    print('Fraction of test samples prediction with a different class')
    print(pred_disagreement)

    print('\n ~~~ Prediction distance ~~~')
    print('Average L2 norm of (prob1 - prob2) in test samples')
    print(pred_distance)