import torch
import numpy as np
import os
import sys
import pdb
from tabulate import tabulate
from avg_index.search_avg import get_avg_model

from helpers.utils import get_folder_name
from helpers.parser import parse_args
from loaders.data import get_data, ROOT_CLUSTER
from model.model import get_model
from robustness_measures.calibration import eval_calibration, ood_gaussian_noise
from robustness_measures.repeatability import eval_repeatability
from helpers.evaluate import eval_ensemble



def _load_model(args, device, seed, opt):
    assert opt in ['SGD', 'EMA_acc', 'EMA_val']

    model = get_model(args, device)
    expt_name = opt + '_' + str(args.lr[0])
    path = get_folder_name(args, expt_name=expt_name, seed=seed)
    ckpt = torch.load(os.path.join(path, 'checkpoint_last.pth.tar'))

    if opt == 'SGD':
        model.load_state_dict(ckpt['state_dict'])
    elif opt in ['EMA_acc', 'EMA_val']:
        alpha = ckpt['best_alpha']
        print(f'Loading EMA with alpha={alpha}')
        model.load_state_dict(ckpt['ema_state_dict_' + str(alpha)])

    return model

def _average_non_zero(arr):
    non_zeros = arr[np.nonzero(arr)]
    return np.round(non_zeros.mean(), 2)

def evaluate_all(args, models, test_loader, device):
    
    results = {}

    # TEST ACCURACY AND LOSS
    _, acc, soft_acc, losses, accs, soft_accs = eval_ensemble(models, test_loader, device)
    # _, avg_model_acc, _, _ = eval_ensemble(models, test_loader, device, avg_model=True)
    print('\n ~~~ Models accuracy ~~~')
    for i in range(len(accs)):
        print(f'Model {i}:\tAccuracy: {accs[i]:.2f} \tLoss: {losses[i]:.4f} \tSoft accuracy: {soft_accs[i]:.2f}')
    print(f'(Prediction) Ensemble Accuracy: {acc:.2f} \tSoft accuracy: {soft_acc:.2f}')

    results['Test Accuracy (%)'] = np.round(np.array(accs).mean(), 2)
    results['Test Loss'] = np.round(np.array(losses).mean(), 2)

    # REPEATABILITY
    disagreement, L2_dist, JS_div = eval_repeatability(args, models, test_loader)
    results['Pred Disagr. (%)'] = _average_non_zero(disagreement)
    results['Pred L2 dist'] = _average_non_zero(L2_dist)
    results['Pred JS div'] = _average_non_zero(JS_div)

    # CALIBRATION
    rms, mad, sf1, test_confidence = eval_calibration(args, models, test_loader)
    results['RMS Calib Error (%)'] = rms
    results['MAD Calib Error (%)'] = mad
    results['Soft F1 Score (%)'] = sf1
    
    # OOD Detection
    ood_gaussian_noise(args, model, test_loader, test_confidence)


    return results

def full_evaluation(args, seeds=[0,1,2]):
    '''
    Evaluate SGD vs EMA solution on mulitple metrics.
    Average of 3 seeds. Always use last model (no early stopping on test set)
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, val_loader, test_loader = get_data(args, args.batch_size[0], args.data_fraction, val_fraction=0)

    # SGD
    print('\n *** Evaluating SGD... ***')
    models = []
    for seed in seeds:
        models.append(_load_model(args, device, seed, opt='SGD'))
    results_SGD = evaluate_all(args, models, test_loader, device)

    # EMA acc
    print('\n *** Evaluating EMA Accuracy... ***')
    models = []
    for seed in seeds:
        models.append(_load_model(args, device, seed, opt='EMA_acc'))
    results_EMA_acc = evaluate_all(args, models, test_loader, device)


    # EMA val
    print('\n *** Evaluating EMA Validation... ***')
    models = []
    for seed in seeds:
        models.append(_load_model(args, device, seed, opt='EMA_val'))
    results_EMA_val = evaluate_all(args, models, test_loader, device)
    

    # Uniform avg of EMA acc
    print('\n *** Evaluating Uniform average of EMA Acc... ***')
    models = []
    for seed in seeds:
        models.append(get_avg_model(args, start=0, end=1, expt_name='EMA_acc_'+str(args.lr[0]), seed=seed))
    results_uniform = evaluate_all(args, models, test_loader, device)

    results = np.vstack((
        np.array([*results_SGD.values()]), 
        np.array([*results_EMA_acc.values()]),
        np.array([*results_EMA_val.values()]),
        np.array([*results_uniform.values()])
        ))
    results_dict = {}
    for i, key in enumerate(results_SGD.keys()):
        results_dict[key] = results[:,i]
    
    print(tabulate([[key, *value] for key, value in results_dict.items()], headers=['', 'SGD (No averaging)', 'EMA Accuracy', 'EMA Validation', 'Uniform (EMA acc)'], tablefmt="pretty"))

if __name__ == '__main__':
    ''' For debugging purposes '''
    args = parse_args()
    full_evaluation(args)

# python full_evaluation.py --net=vgg16 --dataset=cifar100 --lr=0.06