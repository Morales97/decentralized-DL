import torch
import numpy as np
from topology import get_average_model
import torch.nn.functional as F
import pdb

def evaluate_model(model, data_loader, device):
    """Compute loss and accuracy of a single model on a data_loader."""
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            # data = data.to(device)
            output = model(data)
            # output = model(data[None, ...])
            # sum up batch loss
            loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(data_loader.dataset)
    acc = correct / len(data_loader.dataset)

    return loss, acc

def eval_all_models(args, models, test_loader, device):
    acc_workers = []
    loss_workers = []
    for model in models:
        test_loss, acc = evaluate_model(model, test_loader, device)
        acc_workers.append(acc)
        loss_workers.append(test_loss)
    acc = float(np.array(acc_workers).mean()*100)
    test_loss = np.array(loss_workers).mean()

    model = get_average_model(device, models)
    test_loss_avg, acc_avg = evaluate_model(model, test_loader, device)
    
    return acc, test_loss, acc_workers, loss_workers, acc_avg, test_loss_avg


def eval_ensemble(models, test_loader, device, avg_model=False):
    '''
    Evaluate an ensemble of models.    
        if avg_model == True -> average weights
        if avg_model == False -> average prediction probabilities
    '''

    if avg_model:
        model = get_average_model(device, models)
        loss, acc = evaluate_model(model, test_loader, device)
        return loss, acc, None, None

    else:
        corrects = np.zeros(len(models))
        losses = np.zeros(len(models))
        soft_accs = np.zeros(len(models))
        correct = 0
        loss = 0
        soft_acc = 0

        for model in models:
           model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                ensemble_prob = torch.zeros((data.shape[0], model.num_classes)).to(device)

                for i, model in enumerate(models):
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    losses[i] += F.cross_entropy(output, target, reduction='sum').item()
                    corrects[i] += pred.eq(target.view_as(pred)).sum().item()
                    
                    probs = F.softmax(output, dim=1)
                    soft_accs[i] += probs.gather(1, target.unsqueeze(1)).sum().item()   # sum softmax probability of target class
                    ensemble_prob += probs
                
                ensemble_pred = ensemble_prob.argmax(dim=1, keepdim=True)
                correct += ensemble_pred.eq(target.view_as(ensemble_pred)).sum().item()
                soft_acc += ensemble_prob.gather(1, target.unsqueeze(1)).sum().item()/len(models)
        
        accs = []
        for i in range(len(models)):
            accs.append(corrects[i] / len(test_loader.dataset) * 100) 
            losses[i] /= len(test_loader.dataset)
            soft_accs[i] = soft_accs[i] / len(test_loader.dataset) * 100
        acc = correct / len(test_loader.dataset) * 100
        soft_acc = soft_acc / len(test_loader.dataset) * 100

        return None, acc, soft_acc, losses, accs, soft_accs

