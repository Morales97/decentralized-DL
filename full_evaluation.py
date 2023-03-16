import torch
import numpy as np
import os
import sys
import pdb
from tabulate import tabulate
from adversarial.adv_eval import evaluate_adversarial
from avg_index.search_avg import get_avg_model

from helpers.utils import get_folder_name
from helpers.parser import parse_args
from loaders.data import get_data, ROOT_CLUSTER
from model.model import get_model
from robustness_measures.calibration import eval_calibration
from robustness_measures.img_transforms import eval_common_corruptions
from robustness_measures.ood_detection import eval_ood, eval_ood_random_images
from robustness_measures.repeatability import eval_repeatability
from helpers.evaluate import eval_ensemble


def _get_expt_name(args, opt):
    expt_name = opt + '_' + str(args.lr[0])
    if args.label_noise:
        expt_name += '_noise40'
    return expt_name
    
def _load_model(args, device, seed, expt_name, averaging=None, ckpt_name='checkpoint_last.pth.tar'):

    if averaging is None:
        averaging = expt_name
    model = get_model(args, device)
    expt_name = _get_expt_name(args, expt_name)
    path = get_folder_name(args, expt_name=expt_name, seed=seed)
    ckpt = torch.load(os.path.join(path, ckpt_name))

    if averaging == 'SGD':
        model.load_state_dict(ckpt['state_dict'])
    elif averaging in ['EMA_acc', 'EMA_val']:
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
    loss, acc, soft_acc, losses, accs, soft_accs = eval_ensemble(models, test_loader, device)
    # _, avg_model_acc, _, _ = eval_ensemble(models, test_loader, device, avg_model=True)
    print('\n ~~~ Models accuracy ~~~')
    for i in range(len(accs)):
        print(f'Model {i}:\tAccuracy: {accs[i]:.2f} \tLoss: {losses[i]:.4f} \tSoft accuracy: {soft_accs[i]:.2f}')
    print(f'(Prediction) Ensemble Accuracy: {acc:.2f} \tSoft accuracy: {soft_acc:.2f}')

    results['Test Accuracy (%)'] = np.round(np.array(accs).mean(), 2)
    results['Test Loss'] = np.round(np.array(losses).mean(), 2)

    # # REPEATABILITY
    disagreement, L2_dist, JS_div = eval_repeatability(args, models, test_loader)
    results['Pred Disagr. (%)'] = _average_non_zero(disagreement)
    # results['Pred L2 dist'] = _average_non_zero(L2_dist)
    results['Pred JS div'] = _average_non_zero(JS_div)

    # # CALIBRATION
    rms, mad, sf1 = eval_calibration(args, models, test_loader)
    results['RMS Calib Error (%)'] = rms
    results['MAD Calib Error (%)'] = mad
    # results['Soft F1 Score (%)'] = sf1
    
    # # OOD Detection - Anomalous data
    auroc, aupr, fpr = eval_ood(args, models, test_loader)
    # results['FPR (lower better)'] = fpr
    results['AUROC (higher better)'] = auroc
    results['AUPR (higher better)'] = aupr

    # OOD Detection - Random images
    # auroc, aupr, fpr = eval_ood_random_images(args, models, test_loader)
    # results['FPR rand (lower better)'] = fpr
    # results['AUROC rand (higher better)'] = auroc
    # results['AUPR rand (higher better)'] = aupr

    # Common corruptions
    results['Common corruptions (severity=1)'] = eval_common_corruptions(args, models, severities=[1])
    # results['Common corruptions (severities=1-5)'] = eval_common_corruptions(args, models, severities=[1,2,3,4,5])

    # Adversarial attacks
    # results['Adversarial Accuracy (eps=8/255)'] = evaluate_adversarial(args, models, epsilon=8/225)
    results['Adversarial Accuracy (eps=2/255)'] = evaluate_adversarial(args, models, epsilon=2/225)

    return results

def full_evaluation(args, seeds=[0,1,2]):
    '''
    Evaluate SGD vs EMA solution on mulitple metrics.
    Average of 3 seeds. Always use last model (no early stopping on test set)
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, val_loader, test_loader = get_data(args, args.batch_size[0], args.data_fraction, val_fraction=0)

    # SGD
    # print('\n *** Evaluating SGD... ***')
    # models = []
    # for seed in seeds:
    #     models.append(_load_model(args, device, seed, expt_name='SGD'))
    # results_SGD = evaluate_all(args, models, test_loader, device)

    # # EMA acc
    # print('\n *** Evaluating EMA Accuracy... ***')
    # models = []
    # for seed in seeds:
    #     models.append(_load_model(args, device, seed, expt_name='EMA_acc'))
    # results_EMA_acc = evaluate_all(args, models, test_loader, device)


    # # EMA val
    # print('\n *** Evaluating EMA Validation... ***')
    # models = []
    # for seed in seeds:
    #     models.append(_load_model(args, device, seed, expt_name='EMA_val'))
    # results_EMA_val = evaluate_all(args, models, test_loader, device)
    
    # # Uniform avg of SGD
    # print('\n *** Evaluating Uniform average of SGD since epoch 100... ***')
    # models = []
    # for seed in seeds:
    #     models.append(get_avg_model(args, start=0.5, end=1, expt_name=_get_expt_name(args, 'SGD'), seed=seed))
    # results_uniform_sgd = evaluate_all(args, models, test_loader, device)

    # # Uniform avg of EMA acc
    # print('\n *** Evaluating Uniform average of EMA Acc... ***')
    # models = []
    # for seed in seeds:
    #     models.append(get_avg_model(args, start=0, end=1, expt_name=_get_expt_name(args, 'EMA_acc'), seed=seed))
    # results_uniform_acc = evaluate_all(args, models, test_loader, device)

    # # Uniform avg of EMA val
    # print('\n *** Evaluating Uniform average of EMA Val... ***')
    # models = []
    # for seed in seeds:
    #     models.append(get_avg_model(args, start=0, end=1, expt_name=_get_expt_name(args, 'EMA_val'), seed=seed))
    # results_uniform_val = evaluate_all(args, models, test_loader, device)

    print('\n *** Evaluating SGD (train/val)... ***')
    models = []
    for seed in seeds:
        models.append(_load_model(args, device, seed, expt_name='val', averaging='SGD', ckpt_name='checkpoint_last.pth.tar'))
    results_SGD = evaluate_all(args, models, test_loader, device)


    print('\n *** Evaluating EMA Accuracy (train/val)... ***')
    models = []
    for seed in seeds:
        models.append(_load_model(args, device, seed, expt_name='val', averaging='EMA_acc', ckpt_name='best_ema_acc.pth.tar'))
    results_EMA_acc = evaluate_all(args, models, test_loader, device)

    print('\n *** Evaluating EMA Validation (train/val)... ***')
    models = []
    for seed in seeds:
        models.append(_load_model(args, device, seed, expt_name='val', averaging='EMA_val', ckpt_name='best_ema_loss.pth.tar'))
    results_EMA_val = evaluate_all(args, models, test_loader, device)

    results = np.vstack((
        np.array([*results_SGD.values()]), 
        np.array([*results_EMA_acc.values()]),
        np.array([*results_EMA_val.values()]),
        # np.array([*results_uniform_sgd.values()]),
        # np.array([*results_uniform_acc.values()]),
        # np.array([*results_uniform_val.values()])
        ))
    results_dict = {}
    for i, key in enumerate(results_SGD.keys()):
        results_dict[key] = results[:,i]
    
    # print(tabulate([[key, *value] for key, value in results_dict.items()], headers=['', 'SGD (No averaging)', 'EMA Accuracy', 'EMA Validation', 'Uniform (EMA acc)'], tablefmt="pretty"))
    # print(tabulate([[key, *value] for key, value in results_dict.items()], headers=['', 'SGD (No averaging)', 'EMA Accuracy', 'Uniform (EMA val)'], tablefmt="pretty"))
    # print(tabulate([[key, *value] for key, value in results_dict.items()], headers=['', 'SGD (No averaging)', 'EMA Accuracy', 'EMA Validation', 'Uniform (SGD)', 'Uniform (EMA acc)', 'Uniform (EMA val)'], tablefmt="pretty"))
    print(tabulate([[key, *value] for key, value in results_dict.items()], headers=['', 'SGD (No averaging)', 'EMA Accuracy', 'EMA Validation'], tablefmt="pretty"))
    pdb.set_trace()

if __name__ == '__main__':
    ''' For debugging purposes '''
    args = parse_args()
    full_evaluation(args)

# python full_evaluation.py --net=vgg16 --dataset=cifar100 --lr=0.06
# python full_evaluation.py --net=rn18 --dataset=cifar100 --lr=0.8
# python full_evaluation.py --net=widern28 --dataset=cifar100 --lr=0.1