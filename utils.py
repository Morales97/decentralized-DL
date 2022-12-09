import json

from numpy import save

def save_experiment(config, acc=None, test_loss=None, train_loss=None, consensus=None, filename=None):
    if filename is None:
        raise Exception('TODO: make name from config file. Or maybe timestamp')
    
    dicts = {'config': config}


    f = open(filename+'.json', 'w')
    # f.write(json.dumps(config))
    if acc is not None:
        dicts['accuracy'] = acc
    if test_loss is not None:
        dicts['test_loss'] = test_loss
    if train_loss is not None:
        dicts['train_loss'] = train_loss
    if consensus is not None:
        dicts['conensus'] = consensus
    f.write(json.dumps(dicts))
    f.close()


if __name__ == '__main__':
    config = {'test': 1}
    save_experiment(config, None, filename='experiments_mnist/results/test')