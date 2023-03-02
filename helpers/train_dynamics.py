import pdb
import torch
import torch.nn.functional as F
import numpy as np
import os
from sklearn.decomposition import PCA

import sys
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from helpers.parser import SAVE_DIR, parse_args
from model.model import get_model
from loaders.data import get_data

def recursive_glob(rootdir=".", prefix="", suffix=""):
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.startswith(prefix) and filename.endswith(suffix)
    ]

def get_ckpt_steps(ckpt_files):
    steps = [int(file.split('_')[-1][:-8]) for file in ckpt_files]
    root = ckpt_files[0].rsplit('_',1)[0]
    
    return steps, root

def copy_upper_triangle_to_lower(matrix):
    return matrix + matrix.T - np.diag(np.diag(matrix))

@torch.no_grad()
def get_cosine_similarity(model1, model2):
    params1 = torch.cat([p.data.view(-1) for p in model1.parameters()])  # vectorize parameters
    params2 = torch.cat([p.data.view(-1) for p in model2.parameters()])
    return F.cosine_similarity(params1, params2, dim=0)

@torch.no_grad()
def get_prediction_disagreement(model1, model2, loader, device):
    model1.eval()
    model2.eval()
    agree_count = 0
    distance = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output1 = model1(data)
        output2 = model2(data)
        distance += torch.linalg.norm((output1 - output2), dim=1).sum()  # Using L2 norm as distance. Could also use JS

        pred1 = output1.argmax(dim=1, keepdim=True)
        pred2 = output2.argmax(dim=1, keepdim=True)

        agree_count += pred1.eq(pred2).sum().item()
    return distance/len(loader.dataset), 1-agree_count/len(loader.dataset)

@torch.no_grad()
def get_prediction_disagreement_and_correctness(model1, model2, loader, device):
    '''
    also check if predictions agreed/disagreed where correct
    '''
    model1.eval()
    model2.eval()
    agree_count = 0
    distance = 0
    correct_correct = 0
    correct_incorrect = 0
    incorrect_incorrect_same = 0
    incorrect_incorrect_different = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output1 = model1(data)
        output2 = model2(data)
        distance += torch.linalg.norm((output1 - output2), dim=1).sum()  # Using L2 norm as distance. Could also use JS

        pred1 = output1.argmax(dim=1, keepdim=True)
        pred2 = output2.argmax(dim=1, keepdim=True)

        agree_count += pred1.eq(pred2).sum().item()
        
        agreed = pred1.eq(pred2)
        agreed_correct = pred1[agreed].eq(target.view_as(pred1)[agreed]).sum().item()
        correct_correct += agreed_correct
        incorrect_incorrect_same += agreed.sum().item() - agreed_correct

        disagreed = ~pred1.eq(pred2)
        disagreed_correct = pred1[disagreed].eq(target.view_as(pred1)[disagreed]).sum().item() + pred2[disagreed].eq(target.view_as(pred2)[disagreed]).sum().item()
        correct_incorrect += disagreed_correct
        incorrect_incorrect_different += disagreed.sum().item() - disagreed_correct
        pdb.set_trace()

    return distance/len(loader.dataset), 1-agree_count/len(loader.dataset)

def get_train_metrics(args):
    # Get checkpoints of experiment
    ckpt_files = recursive_glob(os.path.join(SAVE_DIR, args.expt_name), prefix='checkpoint')
    ckpt_steps, file_root = get_ckpt_steps(ckpt_files)

    # data
    _, test_loader = get_data(args, batch_size=100)

    # init
    cosine_similarities = np.zeros((len(ckpt_steps), len(ckpt_steps))) 
    pred_disagreement = np.zeros((len(ckpt_steps), len(ckpt_steps)))
    pred_distance = np.zeros((len(ckpt_steps), len(ckpt_steps)))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for i, step_i in enumerate(ckpt_steps):
        ckpt_i = torch.load(os.path.join(SAVE_DIR, file_root + f'_{step_i}.pth.tar'))
        model_i = get_model(args, device)
        model_i.load_state_dict(ckpt_i['state_dict'])

        cosine_similarities[i,i] = 1
        pred_distance[i,i] = 0
        pred_disagreement[i,i] = 0

        for j, step_j in enumerate(ckpt_steps[i+1:]):
            j = j+i+1
            ckpt_j = torch.load(os.path.join(SAVE_DIR, file_root + f'_{step_j}.pth.tar'))
            model_j = get_model(args, device)
            model_j.load_state_dict(ckpt_j['state_dict'])

            cosine_similarities[i,j] = get_cosine_similarity(model_i, model_j)
            pred_distance[i,j], pred_disagreement[i,j] = get_prediction_disagreement(model_i, model_j, test_loader, device)

    # [j,i] = [i,j]
    cosine_similarities = copy_upper_triangle_to_lower(cosine_similarities)
    pred_distance = copy_upper_triangle_to_lower(pred_distance)
    pred_disagreement = copy_upper_triangle_to_lower(pred_disagreement)

    return cosine_similarities, pred_distance, pred_disagreement

def get_pca(args):
    ckpt_files = recursive_glob(os.path.join(SAVE_DIR, args.expt_name), prefix='checkpoint')
    ckpt_steps, file_root = get_ckpt_steps(ckpt_files)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    models = []
    for file in ckpt_files:
        ckpt_i = torch.load(file)
        model_i = get_model(args, device)
        # model_i.load_state_dict(ckpt_i['state_dict'])
        model_i.load_state_dict(ckpt_i['ema_state_dict'])
        models.append(torch.cat([p.data.view(-1) for p in model_i.parameters()]).cpu().numpy())
    models = np.asarray(models)

    pca = PCA(n_components=2)
    models_pca = pca.fit(models).transform(models)
    return models_pca
    
if __name__ == '__main__':
    args = parse_args()

    # models_pca = get_pca(args)
    cos_sim, pred_dist, pred_disag = get_train_metrics(args)

    # np.save(os.path.join(SAVE_DIR, args.expt_name, 'models_pca'), models_pca)
    # np.save(os.path.join(SAVE_DIR, args.expt_name, 'cosine_similarity'), cos_sim)
    # np.save(os.path.join(SAVE_DIR, args.expt_name, 'prediction_distance'), pred_dist)
    # np.save(os.path.join(SAVE_DIR, args.expt_name, 'prediction_disagreement'), pred_disag)
    np.save(os.path.join(SAVE_DIR, args.expt_name, 'cosine_similarity_ema'), cos_sim)
    np.save(os.path.join(SAVE_DIR, args.expt_name, 'prediction_distance_ema'), pred_dist)
    np.save(os.path.join(SAVE_DIR, args.expt_name, 'prediction_disagreement_ema'), pred_disag)


# python helpers/train_dynamics.py --net=convnet_rgb --dataset=cifar10 --expt_name=CNN_lr0.04_decay2
# python helpers/train_dynamics.py --net=rn18 --dataset=cifar100 --expt_name=C4.3_lr0.8

