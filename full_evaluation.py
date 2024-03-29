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
from robustness_measures.model_calibration import eval_calibration, eval_calibration
from robustness_measures.diff_privacy import eval_DP_ranking
from robustness_measures.img_transforms import eval_common_corruptions
from robustness_measures.ood_detection import eval_ood, eval_ood_random_images
from robustness_measures.repeatability import eval_repeatability, eval_repeatability_many
from helpers.evaluate import eval_ensemble, evaluate_model_per_class
from helpers.wa import update_bn


def _get_expt_name(args, opt):
    expt_name = opt + '_' + str(args.lr[0])
    if args.label_noise == '40':
        expt_name += '_noise40'
    elif args.label_noise == 'worse_label':
        expt_name += '_noiseW'
    return expt_name
    
def _load_model(args, device, seed, expt_name, averaging=None, ckpt_name='checkpoint_last.pth.tar', alpha=None, compute_bn=False, train_loader=None, folder_name=None):

    if averaging is None:
        averaging = expt_name
    model = get_model(args, device)
    expt_name = _get_expt_name(args, expt_name)
    if folder_name is None:
        path = get_folder_name(args, expt_name=expt_name, seed=seed)
    else:
        path = os.path.join(args.save_dir, args.dataset, args.net, folder_name)
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

    return model, ckpt['epoch'], alpha

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

def _average_non_zero(arr, round_dec=2):
    non_zeros = arr[np.nonzero(arr)]
    return np.round(non_zeros.mean(), round_dec), np.round(non_zeros.std(), round_dec)


def evaluate_all(args, models, val_loader, test_loader, device, expt_name, averaging, best_per_seed=False):
    
    if not best_per_seed:
        results = _load_saved_results(args, expt_name, averaging)
    else:
        results = _load_saved_results(args, expt_name, averaging + '_best_per_seed')   # potentially mixed between different configs
    # results = {}

    # TEST ACCURACY AND LOSS
    if not 'Test Accuracy (%)' in results.keys():
        # evaluate_model_per_class(models[0], test_loader, device)
        loss, acc, soft_acc, losses, accs, soft_accs = eval_ensemble(models, test_loader, device)
        # _, avg_model_acc, _, _ = eval_ensemble(models, test_loader, device, avg_model=True)
        print('\n ~~~ Models accuracy ~~~')
        for i in range(len(accs)):
            print(f'Model {i}:\tAccuracy: {accs[i]:.2f} \tLoss: {losses[i]:.4f} \tSoft accuracy: {soft_accs[i]:.2f}')
        print(f'(Prediction) Ensemble Accuracy: {acc:.2f} \tSoft accuracy: {soft_acc:.2f}')

        results['Test Accuracy (%)'] = '$' + str(np.round(np.array(accs).mean(), 2)) + ' pm ' + str(np.round(np.array(accs).std(), 2)) + '$'
        results['Test Loss'] = '$' + str(np.round(np.array(losses).mean(), 2)) + ' pm ' + str(np.round(np.array(losses).std(), 2)) + '$'

    # # REPEATABILITY
    # if not 'Pred Disagr. all-to-all (%)' in results.keys():
    #     disagreement = eval_repeatability_many(args, models, test_loader)
    #     results['Pred Disagr. all-to-all (%)'] = disagreement

    if not 'Pred Disagr. (%)' in results.keys():
        disagreement, _, JS_div = eval_repeatability(args, models, test_loader)
        dis_mean, dis_std = _average_non_zero(disagreement)
        js_mean, js_std = _average_non_zero(JS_div, round_dec=3)
        results['Pred Disagr. (%)'] = '$' + str(dis_mean) + ' pm ' + str(dis_std) + '$'
        results['Pred JS div'] = '$' + str(js_mean) + ' pm ' + str(js_std) + '$'

        # results['Pred L2 dist'] = _average_non_zero(L2_dist)

    # # CALIBRATION
    if not 'ECE' in results.keys():
        ece, ece_std, ece_temp, ece_temp_std = eval_calibration(args, models, val_loader, test_loader)
        # results['ECE'] = ece
        results['ECE'] = '$' + str(ece) + ' pm ' + str(ece_std) + '$' 
        results['ECE (Temp. scaling)'] = '$' + str(ece_temp) + ' pm ' + str(ece_temp_std) + '$' 
    
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
    #     # results['Common corruptions (severities=1-5)'] = eval_common_corruptions(args, models, severities=[1,2,3,4,5])

    # # # Adversarial attacks
    if not 'Adversarial Accuracy (eps=2/255)' in results.keys():
        results['Adversarial Accuracy (eps=8/255)'] = evaluate_adversarial(args, models, epsilon=8/225)
        results['Adversarial Accuracy (eps=2/255)'] = evaluate_adversarial(args, models, epsilon=2/225)

    # # # DP ranking membership attack
    # if not 'DP Ranking' in results.keys():
    #     results['DP Ranking'] = eval_DP_ranking(args, models)

    # save results
    expt_name = _get_expt_name(args, expt_name)
    path = get_folder_name(args, expt_name=expt_name, seed=0)   # NOTE using folder with seed=0 by default
    if not best_per_seed:
        with open(os.path.join(path, f'full_eval_results_{averaging}.pkl'), 'wb') as f:
            pickle.dump(results, f)
    else: 
        with open(os.path.join(path, f'full_eval_results_{averaging}_best_per_seed.pkl'), 'wb') as f:
            pickle.dump(results, f)

    return results

def full_evaluation(args, expt_name='val', seeds=[0,1,2]):
    '''
    Evaluate SGD vs EMA solution on mulitple metrics.
    Average of 3 seeds. 
    '''

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, val_loader, test_loader = get_data(args, args.batch_size[0], args.data_fraction, val_fraction=args.val_fraction)

    print('\n *** Evaluating SGD (train/val)... ***')
    models, epochs = [], []
    for seed in seeds:
        if expt_name == 'val':  # use best student accuracy
            model, epoch, _ = _load_model(args, device, seed, expt_name=expt_name, averaging='SGD', ckpt_name='best_student_acc.pth.tar')
        if expt_name == 'SGD':  # use last checkpoint. Reason: cannot use best bc we don't want to eval on Test set, and we discard the "best validation epoch" because we reason that SGD, without implicit regularization, will tend to be always best at last epoch
            model, epoch, _ = _load_model(args, device, seed, expt_name=expt_name, averaging='SGD', ckpt_name='checkpoint_last.pth.tar')
        models.append(model)
        epochs.append(int(epoch))
    results_SGD = evaluate_all(args, models, val_loader, test_loader, device, expt_name=expt_name, averaging='SGD')
    results_SGD['epochs'] = epochs
    results_SGD['EMA decay'] = 0

    print('\n *** Evaluating EMA Accuracy (train/val)... ***')
    models, epochs, alphas = [], [], []
    for seed in seeds:
        if expt_name == 'val':  # use best VAL accuracy
            model, epoch, alpha = _load_model(args, device, seed, expt_name=expt_name, averaging='EMA_acc', ckpt_name='best_ema_acc.pth.tar')
        if expt_name == 'SGD':  # use epoch of best VAL accuracy
            model, epoch, alpha = _load_model(args, device, seed, expt_name=expt_name, averaging='EMA_acc', ckpt_name='ema_acc_epoch.pth.tar')
        models.append(model)
        epochs.append(int(epoch))
        alphas.append(alpha)
    results_EMA_acc = evaluate_all(args, models, val_loader, test_loader, device, expt_name=expt_name, averaging='EMA_acc')
    results_EMA_acc['epochs'] = epochs
    results_EMA_acc['EMA decay'] = alphas

    print('\n *** Evaluating EMA Validation (train/val)... ***')
    models, epochs, alphas = [], [], []
    for seed in seeds:
        if expt_name == 'val':  # use best VAL accuracy
            model, epoch, alpha = _load_model(args, device, seed, expt_name=expt_name, averaging='EMA_val', ckpt_name='best_ema_loss.pth.tar')
        if expt_name == 'SGD':  # use epoch of best VAL accuracy
            model, epoch, alpha = _load_model(args, device, seed, expt_name=expt_name, averaging='EMA_val', ckpt_name='ema_val_epoch.pth.tar')
        models.append(model)
        epochs.append(int(epoch))
        alphas.append(alpha)
    results_EMA_val = evaluate_all(args, models, val_loader, test_loader, device, expt_name=expt_name, averaging='EMA_val')
    results_EMA_val['epochs'] = epochs
    results_EMA_val['EMA decay'] = alphas

    print('\n *** Evaluating EMA Accuracy (alpha=0.998, BN recompute)... ***')
    models, epochs = [], []
    for seed in seeds:
        if expt_name == 'val':
            model, epoch, _ = _load_model(args, device, seed, expt_name=expt_name, averaging='EMA_acc', ckpt_name='best_ema_acc.pth.tar', alpha=0.998, compute_bn=True, train_loader=train_loader)
        if expt_name == 'SGD':
            model, epoch, _ = _load_model(args, device, seed, expt_name=expt_name, averaging='EMA_acc', ckpt_name='ema_acc_epoch.pth.tar', alpha=0.998, compute_bn=True, train_loader=train_loader)
        models.append(model)
        epochs.append(int(epoch))
    results_EMA_acc_BN = evaluate_all(args, models, val_loader, test_loader, device, expt_name=expt_name, averaging='EMA_acc_BN_0.998')
    results_EMA_acc_BN['epochs'] = epochs
    results_EMA_acc_BN['EMA decay'] = 0.998

    print('\n *** Evaluating EMA Validation (alpha=0.998, BN recompute)... ***')
    models, epochs = [], []
    for seed in seeds:
        if expt_name == 'val':
            model, epoch, _ = _load_model(args, device, seed, expt_name=expt_name, averaging='EMA_val', ckpt_name='best_ema_loss.pth.tar', alpha=0.998, compute_bn=True, train_loader=train_loader)
        if expt_name == 'SGD':
            model, epoch, _ = _load_model(args, device, seed, expt_name=expt_name, averaging='EMA_val', ckpt_name='ema_val_epoch.pth.tar', alpha=0.998, compute_bn=True, train_loader=train_loader)
        models.append(model)
        epochs.append(int(epoch))
    results_EMA_val_BN = evaluate_all(args, models, val_loader, test_loader, device, expt_name=expt_name, averaging='EMA_val_BN_0.998')
    results_EMA_val_BN['epochs'] = epochs
    results_EMA_val_BN['EMA decay'] = 0.998

    # # Uniform avg of SGD
    # print('\n *** Evaluating Uniform average of SGD since epoch 100... ***')
    # models = []
    # for seed in seeds:
    #     models.append(get_avg_model(args, start=0.5, end=1, expt_name=_get_expt_name(args, 'SGD'), seed=seed))
    # results_uniform_sgd = evaluate_all(args, models, val_loader, test_loader, device, expt_name=_get_expt_name(args, 'SGD'))

    results = np.vstack((
        np.array([*results_SGD.values()]), 
        np.array([*results_EMA_acc.values()]),
        np.array([*results_EMA_val.values()]),
        np.array([*results_EMA_acc_BN.values()]),
        np.array([*results_EMA_val_BN.values()]),
        # np.array([*results_uniform_sgd.values()])
        ))
    results_dict = {}
    for i, key in enumerate(results_SGD.keys()):
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


    # print(tabulate([[key, *value] for key, value in results_dict.items()], headers=['', 'SGD (No averaging)', 'EMA Accuracy', 'EMA Validation', 'Uniform (SGD)', 'Uniform (EMA acc)', 'Uniform (EMA val)'], tablefmt="pretty"))
    # print(tabulate([[key, *value] for key, value in results_dict.items()], headers=['', 'SGD (No averaging)', 'EMA Accuracy', 'EMA Validation'], tablefmt="pretty"))
    plain_table = tabulate([[key, *value] for key, value in results_dict.items()], headers=['', 'SGD (No averaging)', 'EMA Accuracy', 'EMA Validation', 'EMA Accuracy (BN)', 'EMA Validation (BN)'], tablefmt="pretty")
    plain_table = plain_table.replace('$', ' ')
    print(plain_table)
    
    latex_table = tabulate([[key, *value] for key, value in results_dict.items()], headers=['', 'SGD', 'EMA Acc.', 'EMA Val.', 'EMA Acc. (BN)', 'EMA Val. (BN)'], tablefmt="latex_booktabs")
    latex_table = latex_table.replace('\$', '$')
    latex_table = latex_table.replace('pm', '\pm')
    print(latex_table)
    with open('results_tables.txt', 'a') as f:
        f.write('\n\n----****----\n')
        f.write(f'Net: {args.net}\tDataset: {args.dataset}\tEval on test: {args.eval_on_test}\n')
        f.write(latex_table)

    # print(tabulate([[key, *value] for key, value in results_dict.items()], headers=['', 'EMA Val no pt', 'EMA Val T-IN pt'], tablefmt="pretty"))

# for mixed configs
def full_evaluation_best_per_seed(args, expt_name='val', folder_names=['val_1.2_s0', 'val_0.8_s1', 'val_1.2_s2'], lrs=[1.2, 0.8, 1.2]):
    '''
    Evaluate SGD vs EMA solution on mulitple metrics.
    Select best configuration per seed (e.g., different LRs for different seeds)
    '''

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, val_loader, test_loader = get_data(args, args.batch_size[0], args.data_fraction, val_fraction=args.val_fraction)

    print('\n *** Evaluating SGD (train/val)... ***')
    models, epochs = [], []
    for folder in folder_names:
        if expt_name == 'val':  # use best student accuracy
            model, epoch, _ = _load_model(args, device, None, expt_name=expt_name, averaging='SGD', ckpt_name='best_student_acc.pth.tar', folder_name=folder)
        if expt_name == 'SGD':  # use last checkpoint. Reason: cannot use best bc we don't want to eval on Test set, and we discard the "best validation epoch" because we reason that SGD, without implicit regularization, will tend to be always best at last epoch
            model, epoch, _ = _load_model(args, device, None, expt_name=expt_name, averaging='SGD', ckpt_name='checkpoint_last.pth.tar', folder_name=folder)
        models.append(model)
        epochs.append(int(epoch))
    results_SGD = evaluate_all(args, models, val_loader, test_loader, device, expt_name=expt_name, averaging='SGD', best_per_seed=True)
    results_SGD['epochs'] = epochs
    results_SGD['EMA decay'] = 0
    results_SGD['LR'] = None

    print('\n *** Evaluating EMA Accuracy (train/val)... ***')
    models, epochs, alphas = [], [], []
    for folder in folder_names:
        if expt_name == 'val':  # use best VAL accuracy
            model, epoch, alpha = _load_model(args, device, None, expt_name=expt_name, averaging='EMA_acc', ckpt_name='best_ema_acc.pth.tar', folder_name=folder)
        if expt_name == 'SGD':  # use epoch of best VAL accuracy
            model, epoch, alpha = _load_model(args, device, None, expt_name=expt_name, averaging='EMA_acc', ckpt_name='ema_acc_epoch.pth.tar', folder_name=folder)
        models.append(model)
        epochs.append(int(epoch))
        alphas.append(alpha)
    results_EMA_acc = evaluate_all(args, models, val_loader, test_loader, device, expt_name=expt_name, averaging='EMA_acc', best_per_seed=True)
    results_EMA_acc['epochs'] = epochs
    results_EMA_acc['EMA decay'] = alphas
    results_EMA_acc['LR'] = lrs

    print('\n *** Evaluating EMA Validation (train/val)... ***')
    models, epochs, alphas = [], [], []
    for folder in folder_names:
        if expt_name == 'val':  # use best VAL accuracy
            model, epoch, alpha = _load_model(args, device, None, expt_name=expt_name, averaging='EMA_val', ckpt_name='best_ema_loss.pth.tar', folder_name=folder)
        if expt_name == 'SGD':  # use epoch of best VAL accuracy
            model, epoch, alpha = _load_model(args, device, None, expt_name=expt_name, averaging='EMA_val', ckpt_name='ema_val_epoch.pth.tar', folder_name=folder)
        models.append(model)
        epochs.append(int(epoch))
        alphas.append(alpha)
    results_EMA_val = evaluate_all(args, models, val_loader, test_loader, device, expt_name=expt_name, averaging='EMA_val', best_per_seed=True)
    results_EMA_val['epochs'] = epochs
    results_EMA_val['EMA decay'] = alphas
    results_EMA_val['LR'] = None

    print('\n *** Evaluating EMA Accuracy (alpha=0.998, BN recompute)... ***')
    models, epochs = [], []
    for folder in folder_names:
        if expt_name == 'val':
            model, epoch, _ = _load_model(args, device, None, expt_name=expt_name, averaging='EMA_acc', ckpt_name='best_ema_acc.pth.tar', alpha=0.998, compute_bn=True, train_loader=train_loader, folder_name=folder)
        if expt_name == 'SGD':
            model, epoch, _ = _load_model(args, device, None, expt_name=expt_name, averaging='EMA_acc', ckpt_name='ema_acc_epoch.pth.tar', alpha=0.998, compute_bn=True, train_loader=train_loader, folder_name=folder)
        models.append(model)
        epochs.append(int(epoch))
    results_EMA_acc_BN = evaluate_all(args, models, val_loader, test_loader, device, expt_name=expt_name, averaging='EMA_acc_BN_0.998', best_per_seed=True)
    results_EMA_acc_BN['epochs'] = epochs
    results_EMA_acc_BN['EMA decay'] = 0.998
    results_EMA_acc_BN['LR'] = None

    print('\n *** Evaluating EMA Validation (alpha=0.998, BN recompute)... ***')
    models, epochs = [], []
    for folder in folder_names:
        if expt_name == 'val':
            model, epoch, _ = _load_model(args, device, None, expt_name=expt_name, averaging='EMA_val', ckpt_name='best_ema_loss.pth.tar', alpha=0.998, compute_bn=True, train_loader=train_loader, folder_name=folder)
        if expt_name == 'SGD':
            model, epoch, _ = _load_model(args, device, None, expt_name=expt_name, averaging='EMA_val', ckpt_name='ema_val_epoch.pth.tar', alpha=0.998, compute_bn=True, train_loader=train_loader, folder_name=folder)
        models.append(model)
        epochs.append(int(epoch))
    results_EMA_val_BN = evaluate_all(args, models, val_loader, test_loader, device, expt_name=expt_name, averaging='EMA_val_BN_0.998', best_per_seed=True)
    results_EMA_val_BN['epochs'] = epochs
    results_EMA_val_BN['EMA decay'] = 0.998
    results_EMA_val_BN['LR'] = None

    # # Uniform avg of SGD
    # print('\n *** Evaluating Uniform average of SGD since epoch 100... ***')
    # models = []
    # for seed in seeds:
    #     models.append(get_avg_model(args, start=0.5, end=1, expt_name=_get_expt_name(args, 'SGD'), seed=seed))
    # results_uniform_sgd = evaluate_all(args, models, val_loader, test_loader, device, expt_name=_get_expt_name(args, 'SGD'))

    results = np.vstack((
        np.array([*results_SGD.values()]), 
        np.array([*results_EMA_acc.values()]),
        np.array([*results_EMA_val.values()]),
        np.array([*results_EMA_acc_BN.values()]),
        np.array([*results_EMA_val_BN.values()]),
        # np.array([*results_uniform_sgd.values()])
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
    print(tabulate([[key, *value] for key, value in results_dict.items()], headers=['', 'SGD', 'EMA Acc.', 'EMA Val.', 'EMA Acc. (BN)', 'EMA Val. (BN)'], tablefmt="latex_booktabs"))
    


if __name__ == '__main__':
    ''' For debugging purposes '''
    args = parse_args()
    
    if args.eval_on_test:
        expt_name = 'SGD'   
    else:
        expt_name = 'val'   # NOTE first part of experiment name. this will eval models from folders 'val_[lr]_s*'

    # DEFAULT
    full_evaluation(args, expt_name)

    # RN-18, C-100, best configs
    # full_evaluation_best_per_seed(args, expt_name, folder_names=['val_1.2_s0', 'val_0.8_s1', 'val_1.2_s2'], lrs=[1.2, 0.8, 1.2])  # NOTE for mixed configs
    # full_evaluation_best_per_seed(args, expt_name, folder_names=['SGD_1.2_s0', 'SGD_0.8_s1', 'SGD_1.2_s2'], lrs=[1.2, 0.8, 1.2])  # NOTE for mixed configs. need to set expt_name=SGD as well

    # RN-18, C-10, best configs
    # full_evaluation_best_per_seed(args, expt_name, folder_names=['val_0.4_s0', 'val_0.8_s1', 'val_0.4_s2'], lrs=[0.4, 0.8, 0.4])  # NOTE for mixed configs
    # full_evaluation_best_per_seed(args, expt_name, folder_names=['SGD_0.4_s0', 'SGD_0.8_s1', 'SGD_0.4_s2'], lrs=[0.4, 0.8, 0.4])  # NOTE for mixed configs. need to set expt_name=SGD as well


# python full_evaluation.py --net=vgg16 --dataset=cifar100 --lr=0.06 --eval_on_test=True    # NOTE set to False to see 'val' results
# python full_evaluation.py --net=rn18 --dataset=cifar100 --lr=0.8 --eval_on_test=True
# python full_evaluation.py --net=rn18 --dataset=cifar10 --lr=0.4 --eval_on_test=False
# python full_evaluation.py --net=rn34 --dataset=cifar100 --lr=0.8 --label_noise=40 --eval_on_test=False
# python full_evaluation.py --net=rn34 --dataset=cifar10 --lr=0.8 --label_noise=worse_label --eval_on_test=True
# python full_evaluation.py --net=widern28 --dataset=cifar100 --lr=0.1 --eval_on_test=False
# python full_evaluation.py --net=rn18 --dataset=tiny-in --lr=0.8 --eval_on_test=False

