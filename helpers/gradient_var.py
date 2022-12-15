import numpy as np
import pdb
import time 
import torch
import torch.nn as nn
import torch.nn.functional as F

###################### VARIANCE ##########################

def compute_stochastic_gradient_variance(config, model, opt, train_loader_iter, device):
    
    gradients = []
    for step in range(50):
        input, target = next(train_loader_iter)
        input = input.to(device)
        target = target.to(device)

        model.train()
        output = model(input)
        opt.zero_grad()
        loss = F.nll_loss(output, target)
        loss.backward()
        # opt.step() NOTE we don't optimize the model!

        grad = []
        for param in model.parameters():
            # grad.append(param.grad.numpy())
            grad.append(param.grad.numpy().flatten())
        gradients.append(np.concatenate([*grad]))
    gradients = np.array(gradients)
    variance = gradients.var(axis=0).mean()
    
    return variance

def compute_node_gradient_variance(config, models, opts, loader_iter, device):
    
    gradients = []
    input, target = next(loader_iter)
    input = input.to(device)
    target = target.to(device)
    for i in range(len(models)):
        models[i].train()
        output = models[i](input)
        opts[i].zero_grad()
        loss = F.nll_loss(output, target)
        loss.backward()
        # opt.step() NOTE we don't optimize the model!

        grad = []
        for param in models[i].parameters():
            grad.append(param.grad.numpy().flatten())
        gradients.append(np.concatenate([*grad]))
    gradients = np.array(gradients)
    variance = gradients.var(axis=0).mean()

    return variance

def compute_gradient_variance_ratio(config, models, opts, train_loader_iter, train_loader, test_loader, device):

    stochastic_var = []
    for i in range(len(models)):
        train_loader_iter[i] = iter(train_loader[i])
        var = compute_stochastic_gradient_variance(config, models[i], opts[i], train_loader_iter[i], device)
        stochastic_var.append(var)
    
    node_var = []
    for i in range(50):
        # loader_iter = iter(train_loader[0])
        loader_iter = iter(test_loader)
        var = compute_node_gradient_variance(config, models, opts, loader_iter, device)  # NOTE using only one train loader!
        node_var.append(var)

    mean_stoch_var = np.array(stochastic_var).mean()
    mean_node_var = np.array(node_var).mean()
    ratio = mean_stoch_var / mean_node_var

    return np.array([ratio, mean_stoch_var, mean_node_var])

###################### COHERENCE ##########################


def compute_stochastic_gradient_coherence(config, model, opt, train_loader_iter, device):
    
    gradients = []
    for step in range(50):
        input, target = next(train_loader_iter)
        input = input.to(device)
        target = target.to(device)

        model.train()
        output = model(input)
        opt.zero_grad()
        loss = F.nll_loss(output, target)
        loss.backward()
        # opt.step() NOTE we don't optimize the model!

        grad = []
        for param in model.parameters():
            # grad.append(param.grad.numpy())
            grad.append(param.grad.numpy().flatten())
        gradients.append(np.concatenate([*grad]))
    gradients = np.array(gradients)
    mean_grad = gradients.mean(axis=0)
    coherence = []
    for i in range(gradients.shape[0]):
        num = mean_grad @ gradients[i]
        coherence.append(num / (gradients[i]@ gradients[i]))
    
    return np.mean(coherence)

def compute_stochastic_gradient_coherence_iid(config, model, opt, train_loader, device):
    
    gradients = []
    for step in range(50):
        input, target = next(iter(train_loader))
        input = input.to(device)
        target = target.to(device)

        model.train()
        output = model(input)
        opt.zero_grad()
        loss = F.nll_loss(output, target)
        loss.backward()
        # opt.step() NOTE we don't optimize the model!

        grad = []
        for param in model.parameters():
            # grad.append(param.grad.numpy())
            grad.append(param.grad.numpy().flatten())
        gradients.append(np.concatenate([*grad]))
    gradients = np.array(gradients)
    mean_grad = gradients.mean(axis=0)
    coherence = []
    for i in range(gradients.shape[0]):
        num = mean_grad @ gradients[i]
        coherence.append(num / (gradients[i]@ gradients[i]))
    
    return np.mean(coherence)

def compute_node_gradient_coherence(config, models, opts, loader_iter, device):
    
    gradients = []
    input, target = next(loader_iter)
    input = input.to(device)
    target = target.to(device)
    for i in range(len(models)):
        models[i].train()
        output = models[i](input)
        opts[i].zero_grad()
        loss = F.nll_loss(output, target)
        loss.backward()
        # opt.step() NOTE we don't optimize the model!

        grad = []
        for param in models[i].parameters():
            # grad.append(param.grad.numpy())
            grad.append(param.grad.numpy().flatten())
        gradients.append(np.concatenate([*grad]))
    gradients = np.array(gradients)
    mean_grad = gradients.mean(axis=0)
    coherence = []
    for i in range(gradients.shape[0]):
        num = mean_grad @ gradients[i]
        coherence.append(num / (gradients[i] @ gradients[i]))
    
    mean_coherence = np.array(coherence).mean()

    return mean_coherence


def compute_gradient_coherence_ratio(config, models, opts, train_loader_iter, train_loader, test_loader, device):

    stochastic_coh = []
    for i in range(len(models)):
        train_loader_iter[i] = iter(train_loader[i])
        coh = compute_stochastic_gradient_coherence(config, models[i], opts[i], train_loader_iter[i], device)
        stochastic_coh.append(coh)
    
    node_coh = []
    for i in range(50):
        # loader_iter = iter(train_loader[0])
        loader_iter = iter(test_loader)
        coh = compute_node_gradient_coherence(config, models, opts, loader_iter, device)  # NOTE using only one train loader!
        node_coh.append(coh)

    mean_stoch_coh = np.array(stochastic_coh).mean()
    mean_node_coh = np.array(node_coh).mean()
    ratio = mean_stoch_coh / mean_node_coh

    return np.array([ratio, mean_stoch_coh, mean_node_coh])

def compute_gradient_coherence_ratio_iid(config, models, opts, train_loader, device):

    stochastic_coh = []
    for i in range(len(models)):
        coh = compute_stochastic_gradient_coherence_iid(config, models[i], opts[i], train_loader, device)
        stochastic_coh.append(coh)
    
    node_coh = []
    for i in range(50):
        train_loader_iter = iter(train_loader)
        coh = compute_node_gradient_coherence(config, models, opts, train_loader_iter, device)  # NOTE using only one train loader!
        node_coh.append(coh)

    mean_stoch_coh = np.array(stochastic_coh).mean()
    mean_node_coh = np.array(node_coh).mean()
    ratio = mean_stoch_coh / mean_node_coh

    return np.array([ratio, mean_stoch_coh, mean_node_coh])

###################### VARIANCE and COHERENCE ##########################

def stochastic_var_and_coh(config, model, opt, train_loader_iter, device):
    ''' Compute Stochastic gradient variance and coherence '''
    gradients = []
    for step in range(50):
        input, target = next(train_loader_iter)
        input = input.to(device)
        target = target.to(device)

        model.train()
        output = model(input)
        opt.zero_grad()
        loss = F.nll_loss(output, target)
        loss.backward()
        # opt.step() NOTE we don't optimize the model!

        grad = []
        for param in model.parameters():
            # grad.append(param.grad.numpy())
            grad.append(param.grad.numpy().flatten())
        gradients.append(np.concatenate([*grad]))
    gradients = np.array(gradients)

    # variance
    variances = gradients.var(axis=0)
    variance = np.mean(variances)

    #coherence
    mean_grad = gradients.mean(axis=0)
    coherences = []
    for i in range(gradients.shape[0]):
        num = mean_grad @ gradients[i]
        coherences.append(num / (gradients[i]@ gradients[i]))
    coherence = np.mean(coherences)

    return variance, coherence


def node_var_and_coh(config, models, opts, loader_iter, device):
    ''' Compute Node gradient variance and coherence '''
    gradients = []
    input, target = next(loader_iter)
    input = input.to(device)
    target = target.to(device)
    for i in range(len(models)):
        models[i].train()
        output = models[i](input)
        opts[i].zero_grad()
        loss = F.nll_loss(output, target)
        loss.backward()
        # opt.step() NOTE we don't optimize the model!

        grad = []
        for param in models[i].parameters():
            grad.append(param.grad.numpy().flatten())
        gradients.append(np.concatenate([*grad]))
    gradients = np.array(gradients)
    
    # variance
    variance = gradients.var(axis=0)
    variance = np.mean(variance)

    # coherence
    mean_grad = gradients.mean(axis=0)
    coherence = []
    for i in range(gradients.shape[0]):
        num = mean_grad @ gradients[i]
        coherence.append(num / (gradients[i] @ gradients[i]))
    coherence = np.mean(coherence)

    return variance, coherence

def compute_var_and_coh(config, models, opts, train_loader_iter, train_loader, test_loader, device):

    stochastic_var = []
    stochastic_coh = []
    for i in range(len(models)):
        train_loader_iter[i] = iter(train_loader[i])
        var, coh = stochastic_var_and_coh(config, models[i], opts[i], train_loader_iter[i], device)
        stochastic_var.append(var)
        stochastic_coh.append(coh)
    
    node_var = []
    node_coh = []
    for i in range(50):
        # loader_iter = iter(train_loader[0])
        loader_iter = iter(test_loader)
        var, coh = node_var_and_coh(config, models, opts, loader_iter, device)  # NOTE using only one train loader!
        node_var.append(var)
        node_coh.append(coh)

    mean_stoch_var = np.mean(stochastic_var)
    mean_node_var = np.mean(node_var)
    ratio_var = mean_stoch_var / mean_node_var
    print('Variance Stoch/Node: %.3f\tStochastic: %.6f\tNode:%.6f' % (ratio_var, mean_stoch_var, mean_node_var))
    print(np.var(stochastic_var))
    print(np.var(node_var))
    mean_stoch_coh = np.mean(stochastic_coh)
    mean_node_coh = np.mean(node_coh)
    ratio_coh = mean_stoch_coh / mean_node_coh
    print('Coherence Stoch/Node: %.3f\tStochastic: %.6f\tNode:%.6f' % (ratio_coh, mean_stoch_coh, mean_node_coh))
    
    return np.array([ratio_var, mean_stoch_var, mean_node_var]), np.array([ratio_coh, mean_stoch_coh, mean_node_coh])