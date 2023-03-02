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

    else:
        corrects = np.zeros(len(models))
        losses = np.zeros(len(models))
        correct = 0
        loss = 0

        for model in models:
           model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                ensemble_output = torch.zeros((data.shape[0], model.num_classes))

                for i, model in enumerate(models):
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    losses[i] += F.cross_entropy(output, target, reduction='sum').item()
                    corrects[i] += pred.eq(target.view_as(pred)).sum().item()

                    ensemble_output += F.softmax(output)
                    ensemble_pred = ensemble_output.argmax(dim=1, keepdim=True)
                    loss += F.cross_entropy(ensemble_output, ensemble_pred, reduction='sum').item()
                    correct += ensemble_pred.eq(target.view_as(ensemble_pred)).sum().item()
                    pdb.set_trace()
