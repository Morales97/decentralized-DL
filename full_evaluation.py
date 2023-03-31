from pyrsistent import v
import torch
import numpy as np
import os
import sys
import pdb
import pickle
from tabulate import tabulate
from adversarial.adv_eval import evaluate_adversarial
from avg_index.search_avg import get_avg_model

from helpers.utils import get_folder_name
from helpers.parser import parse_args
from loaders.data import get_data, ROOT_CLUSTER
from model.model import get_model
from robustness_measures.model_calibration import eval_calibration, eval_calibration_new
from robustness_measures.diff_privacy import eval_DP_ranking
from robustness_measures.img_transforms import eval_common_corruptions
from robustness_measures.ood_detection import eval_ood, eval_ood_random_images
from robustness_measures.repeatability import eval_repeatability, eval_repeatability_many
from helpers.evaluate import eval_ensemble, evaluate_model_per_class
from helpers.wa import update_bn


def _get_expt_name(args, opt):
    expt_name = opt + '_' + str(args.lr[0])
    if args.label_noise:
        expt_name += '_noise40'
    return expt_name
    
def _load_model(args, device, seed, expt_name, averaging=None, ckpt_name='checkpoint_last.pth.tar', alpha=None, compute_bn=False, train_loader=None):

    if averaging is None:
        averaging = expt_name
    model = get_model(args, device)
    expt_name = _get_expt_name(args, expt_name)
    path = get_folder_name(args, expt_name=expt_name, seed=seed)
    ckpt = torch.load(os.path.join(path, ckpt_name))

    if averaging == 'SGD':
        model.load_state_dict(ckpt['state_dict'])
    elif averaging in ['EMA_acc', 'EMA_val']:
        if alpha is None:
            alpha = ckpt['best_alpha']
        print(f'Loading EMA with alpha={alpha}')
        model.load_state_dict(ckpt['ema_state_dict_' + str(alpha)])

    if compute_bn:
        update_bn(args, train_loader, model, device)

    return model

def _load_saved_results(args, expt_name, averaging, seed=0):
    ''' Load previously computed results. Results are usually for seed=[0,1,2], but saved in folder of seed=0 by defualt '''
    expt_name = _get_expt_name(args, expt_name)
    path = get_folder_name(args, expt_name=expt_name, seed=seed)
    file_path = os.path.join(path, f'full_eval_results_{averaging}.pkl')
    if not os.path.exists(file_path):
        return {}
    
    with open(os.path.join(path, f'full_eval_results_{averaging}.pkl'), 'rb') as f:
        results = pickle.load(f)
        print(f'Loaded results for {results.keys()}')
    return results

def _average_non_zero(arr):
    non_zeros = arr[np.nonzero(arr)]
    return np.round(non_zeros.mean(), 2)

def evaluate_all(args, models, val_loader, test_loader, device, expt_name, averaging):
    
    results = _load_saved_results(args, expt_name, averaging)
    
    # TEST ACCURACY AND LOSS
    if not 'Test Accuracy (%)' in results.keys():
        # evaluate_model_per_class(models[0], test_loader, device)
        loss, acc, soft_acc, losses, accs, soft_accs = eval_ensemble(models, test_loader, device)
        # _, avg_model_acc, _, _ = eval_ensemble(models, test_loader, device, avg_model=True)
        print('\n ~~~ Models accuracy ~~~')
        for i in range(len(accs)):
            print(f'Model {i}:\tAccuracy: {accs[i]:.2f} \tLoss: {losses[i]:.4f} \tSoft accuracy: {soft_accs[i]:.2f}')
        print(f'(Prediction) Ensemble Accuracy: {acc:.2f} \tSoft accuracy: {soft_acc:.2f}')

        results['Test Accuracy (%)'] = np.round(np.array(accs).mean(), 2)
        results['Test Loss'] = np.round(np.array(losses).mean(), 2)

    # # REPEATABILITY
    # if not 'Pred Disagr. all-to-all (%)' in results.keys():
    #     disagreement = eval_repeatability_many(args, models, test_loader)
    #     results['Pred Disagr. all-to-all (%)'] = disagreement

    if not 'Pred Disagr. (%)' in results.keys():
        disagreement, L2_dist, JS_div = eval_repeatability(args, models, test_loader)
        results['Pred Disagr. (%)'] = _average_non_zero(disagreement)
        results['Pred L2 dist'] = _average_non_zero(L2_dist)
        results['Pred JS div'] = _average_non_zero(JS_div)

    # # CALIBRATION
    if not 'ECE (Temp. scaling)' in results.keys():
        mce, ece, mce_temp, ece_temp, mce_binner, ece_binner = eval_calibration_new(args, models, val_loader, test_loader)
        results['ECE'] = ece
        results['ECE (Temp. scaling)'] = ece_temp
        
        # results['MCE'] = mce
        # results['MCE (Temp. scaling)'] = mce_temp
        # results['MCE (Binner scaling)'] = mce_binner
        # results['ECE (Binner scaling)'] = ece_binner

        # rms, mad, sf1 = eval_calibration(args, models, test_loader)
        # results['RMS Calib Error (%)'] = rms
        # results['MAD Calib Error (%)'] = mad
        # # results['Soft F1 Score (%)'] = sf1
    
    # # OOD Detection - Anomalous data
    # auroc, aupr, fpr = eval_ood(args, models, test_loader)
    # # results['FPR (lower better)'] = fpr
    # results['AUROC (higher better)'] = auroc
    # results['AUPR (higher better)'] = aupr

    # OOD Detection - Random images
    # auroc, aupr, fpr = eval_ood_random_images(args, models, test_loader)
    # results['FPR rand (lower better)'] = fpr
    # results['AUROC rand (higher better)'] = auroc
    # results['AUPR rand (higher better)'] = aupr

    # # Common corruptions
    if not 'Common corruptions (severity=1)' in results.keys():
        results['Common corruptions (severity=1)'] = eval_common_corruptions(args, models, severities=[1])
        # results['Common corruptions (severities=1-5)'] = eval_common_corruptions(args, models, severities=[1,2,3,4,5])

    # # Adversarial attacks
    if not 'Adversarial Accuracy (eps=2/255)' in results.keys():
        results['Adversarial Accuracy (eps=8/255)'] = evaluate_adversarial(args, models, epsilon=8/225)
        results['Adversarial Accuracy (eps=2/255)'] = evaluate_adversarial(args, models, epsilon=2/225)

    # # DP ranking membership attack
    if not 'DP Ranking' in results.keys():
        results['DP Ranking'] = eval_DP_ranking(args, models)

    # save results
    expt_name = _get_expt_name(args, expt_name)
    path = get_folder_name(args, expt_name=expt_name, seed=0)   # NOTE using folder with seed=0 by default
    with open(os.path.join(path, f'full_eval_results_{averaging}.pkl'), 'wb') as f:
        pickle.dump(results, f)

    return results

def full_evaluation(args, seeds=[0,1,2]):
    '''
    Evaluate SGD vs EMA solution on mulitple metrics.
    Average of 3 seeds. Always use last model (no early stopping on test set)
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, val_loader, test_loader = get_data(args, args.batch_size[0], args.data_fraction, val_fraction=args.val_fraction)

    # SGD
    # print('\n *** Evaluating SGD... ***')
    # models = []
    # for seed in seeds:
    #     models.append(_load_model(args, device, seed, expt_name='SGD'))
    # results_SGD = evaluate_all(args, models, val_loader, test_loader, device, expt_name='SGD')

    # # EMA acc
    # print('\n *** Evaluating EMA Accuracy... ***')
    # models = []
    # for seed in seeds:
    #     models.append(_load_model(args, device, seed, expt_name='EMA_acc'))
    # results_EMA_acc = evaluate_all(args, models, val_loader, test_loader, device, expt_name='EMA_acc')


    # # EMA val
    # print('\n *** Evaluating EMA Validation... ***')
    # models = []
    # for seed in seeds:
    #     models.append(_load_model(args, device, seed, expt_name='EMA_val'))
    # results_EMA_val = evaluate_all(args, models, val_loader, test_loader, device, expt_name='EMA_val')
    
    # # Uniform avg of SGD
    # print('\n *** Evaluating Uniform average of SGD since epoch 100... ***')
    # models = []
    # for seed in seeds:
    #     models.append(get_avg_model(args, start=0.5, end=1, expt_name=_get_expt_name(args, 'SGD'), seed=seed))
    # results_uniform_sgd = evaluate_all(args, models, val_loader, test_loader, device, expt_name=_get_expt_name(args, 'SGD'))

    # # Uniform avg of EMA acc
    # print('\n *** Evaluating Uniform average of EMA Acc... ***')
    # models = []
    # for seed in seeds:
    #     models.append(get_avg_model(args, start=0, end=1, expt_name=_get_expt_name(args, 'EMA_acc'), seed=seed))
    # results_uniform_acc = evaluate_all(args, models, val_loader, test_loader, device, expt_name=_get_expt_name(args, 'EMA_acc'))

    # # Uniform avg of EMA val
    # print('\n *** Evaluating Uniform average of EMA Val... ***')
    # models = []
    # for seed in seeds:
    #     models.append(get_avg_model(args, start=0, end=1, expt_name=_get_expt_name(args, 'EMA_val'), seed=seed))
    # results_uniform_val = evaluate_all(args, models, val_loader, test_loader, device, expt_name=_get_expt_name(args, 'EMA_val'))

    print('\n *** Evaluating SGD (train/val)... ***')
    models = []
    for seed in seeds:
        models.append(_load_model(args, device, seed, expt_name='val', averaging='SGD', ckpt_name='checkpoint_last.pth.tar'))
    results_SGD = evaluate_all(args, models, val_loader, test_loader, device, expt_name='val', averaging='SGD')

    print('\n *** Evaluating EMA Accuracy (train/val)... ***')
    models = []
    for seed in seeds:
        models.append(_load_model(args, device, seed, expt_name='val', averaging='EMA_acc', ckpt_name='best_ema_acc.pth.tar'))
    results_EMA_acc = evaluate_all(args, models, val_loader, test_loader, device, expt_name='val', averaging='EMA_acc')

    print('\n *** Evaluating EMA Validation (train/val)... ***')
    models = []
    for seed in seeds:
        models.append(_load_model(args, device, seed, expt_name='val', averaging='EMA_val', ckpt_name='best_ema_loss.pth.tar'))
    results_EMA_val = evaluate_all(args, models, val_loader, test_loader, device, expt_name='val', averaging='EMA_val')

    print('\n *** Evaluating EMA Accuracy (alpha=0.998, BN recompute)... ***')
    models = []
    for seed in seeds:
        # models = None
        models.append(_load_model(args, device, seed, expt_name='val', averaging='EMA_acc', ckpt_name='best_ema_acc.pth.tar', alpha=0.998, compute_bn=True, train_loader=train_loader))
    results_EMA_acc_BN = evaluate_all(args, models, val_loader, test_loader, device, expt_name='val', averaging='EMA_acc_BN_0.998')

    print('\n *** Evaluating EMA Validation (alpha=0.998, BN recompute)... ***')
    models = []
    for seed in seeds:
        # models = None
        models.append(_load_model(args, device, seed, expt_name='val', averaging='EMA_val', ckpt_name='best_ema_loss.pth.tar', alpha=0.998, compute_bn=True, train_loader=train_loader))
    results_EMA_val_BN = evaluate_all(args, models, val_loader, test_loader, device, expt_name='val', averaging='EMA_val_BN_0.998')

    for key in results_SGD.keys():
        if key not in results_EMA_acc_BN.keys():
            results_EMA_acc_BN[key] = 0
            results_EMA_val_BN[key] = 0

    for key in results_EMA_acc_BN.keys():
        if key not in results_SGD.keys():
            results_EMA_acc_BN.pop(key)
            results_EMA_val_BN.pop(key)

    results = np.vstack((
        np.array([*results_SGD.values()]), 
        np.array([*results_EMA_acc.values()]),
        np.array([*results_EMA_val.values()]),
        np.array([*results_EMA_acc_BN.values()]),
        np.array([*results_EMA_val_BN.values()]),
        # np.array([*results_uniform_sgd.values()]),
        # np.array([*results_uniform_acc.values()]),
        # np.array([*results_uniform_val.values()])
        ))
    results_dict = {}
    for i, key in enumerate(results_EMA_acc.keys()):
        results_dict[key] = results[:,i]
    
    # Drop keys to show only desired metrics
    results_dict.pop('Pred Disagr. all-to-all (%)', None)
    # results_dict.pop('Pred Disagr. (%)', None)
    # results_dict.pop('Pred JS div', None)
    results_dict.pop('Pred L2 dist', None)
    # results_dict.pop('ECE', None)
    # results_dict.pop('ECE (Temp. scaling)', None)
    # results_dict.pop('Common corruptions (severity=1)', None)
    # results_dict.pop('Adversarial Accuracy (eps=2/255)', None)
    # results_dict.pop('DP Ranking', None)

    # drop deprecated keys (which may still be in old saved results)
    results_dict.pop('RMS Calib error', None)
    results_dict.pop('RMS top-label Calib error', None)
    results_dict.pop('MCE', None)
    results_dict.pop('MCE (Temp. scaling)', None)
    results_dict.pop('MCE (Binner scaling)', None)
    results_dict.pop('ECE (Binner scaling)', None)

    # print(tabulate([[key, *value] for key, value in results_dict.items()], headers=['', 'SGD (No averaging)', 'EMA Accuracy', 'EMA Validation', 'Uniform (SGD)', 'Uniform (EMA acc)', 'Uniform (EMA val)'], tablefmt="pretty"))
    # print(tabulate([[key, *value] for key, value in results_dict.items()], headers=['', 'SGD (No averaging)', 'EMA Accuracy', 'EMA Validation'], tablefmt="pretty"))
    print(tabulate([[key, *value] for key, value in results_dict.items()], headers=['', 'SGD (No averaging)', 'EMA Accuracy', 'EMA Validation', 'EMA Accuracy (BN)', 'EMA Validation (BN)'], tablefmt="pretty"))
    # print(tabulate([[key, *value] for key, value in results_dict.items()], headers=['', 'EMA Val no pt', 'EMA Val T-IN pt'], tablefmt="pretty"))

if __name__ == '__main__':
    ''' For debugging purposes '''
    args = parse_args()
    full_evaluation(args)

# python full_evaluation.py --net=vgg16 --dataset=cifar100 --lr=0.06
# python full_evaluation.py --net=rn18 --dataset=cifar100 --lr=0.8
# python full_evaluation.py --net=rn18 --dataset=cifar10 --lr=0.4
# python full_evaluation.py --net=rn18 --dataset=cifar100 --lr=0.8 --label_noise
# python full_evaluation.py --net=widern28 --dataset=cifar100 --lr=0.1
# python full_evaluation.py --net=rn18 --dataset=tiny-in --lr=0.8

# python full_evaluation.py --net=rn18 --dataset=cifar100 --lr=0.8