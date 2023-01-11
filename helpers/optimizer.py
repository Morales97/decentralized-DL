import torch.optim as optim


def get_optimizer(args, model):
    if args.opt == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=args.nesterov, weight_decay=args.wd)
    else:
        raise Exception('Optimizer not supported')
    
    return optimizer