import torch
import torch.nn.functional as F

import os
import sys
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from adversarial import attacks
from helpers.parser import parse_args
from loaders.data import get_data, ROOT_CLUSTER
from model.model import get_model
from avg_index.search_avg import find_index_ckpt
from avg_index.avg_index import UniformAvgIndex, ModelAvgIndex, TriangleAvgIndex
from torchvision import datasets, transforms
import torch.utils.data as data
import pdb

def evaluate(model, test_loader, adv=True, epsilon=8./255):
    adversary = attacks.PGD_linf(epsilon=epsilon, num_steps=20, step_size=2./255).cuda()

    model.eval()
    if adv is False:
        torch.set_grad_enabled(False)
    running_loss = 0
    running_acc = 0
    count = 0
    for i, batch in enumerate(test_loader):
        bx = batch[0].cuda()
        by = batch[1].cuda()

        count += by.size(0)

        adv_bx = adversary(model, bx, by) if adv else bx
        with torch.no_grad():
            pdb.set_trace()
            logits = model(adv_bx) # TODO change this
            # logits = model(adv_bx * 2 - 1) # TODO change this

        loss = F.cross_entropy(logits.data, by, reduction='sum')
        running_loss += loss.cpu().data.numpy()
        running_acc += (torch.max(logits, dim=1)[1] == by).float().sum(0).cpu().data.numpy()
    running_loss /= count
    running_acc /= count

    loss = running_loss
    acc = running_acc

    if adv is False:
        torch.set_grad_enabled(True)
    return loss, acc


if __name__ == '__main__':
    args = parse_args()

    # _, test_loader = get_data(args, args.batch_size[0], args.data_fraction)
    dataset_loader = datasets.CIFAR100
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = dataset_loader(
        root=ROOT_CLUSTER,
        train=False,
        transform=transform,
        download=True,
    )
    test_loader = data.DataLoader(dataset, batch_size=200, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = os.path.join(args.save_dir, args.expt_name)
    index_ckpt_file, step = find_index_ckpt(save_dir)
    state_dir = os.path.join(save_dir, index_ckpt_file)

    _index = UniformAvgIndex('.')
    state_dict = torch.load(state_dir)
    _index.load_state_dict(state_dict)

    index = ModelAvgIndex(
            get_model(args, device),              # NOTE only supported with solo mode now.
            _index,
            include_buffers=True,
        )
    
    av_ckpts = list(state_dict['available_checkpoints'])
    av_ckpts.sort()
    model = index.avg_from(av_ckpts[int(3*len(av_ckpts)//6)], until=av_ckpts[-1])   # take as model the avg between half and end of training for now

    # epsilon = 8./255
    # loss, acc = evaluate(model, test_loader, epsilon=epsilon)
    # print(f'Adversarial Test Accuracy (eps={epsilon}): {acc} \t Advesarial Test Loss: {loss}')

    # epsilon = 4./255
    # loss, acc = evaluate(model, test_loader, epsilon=epsilon)
    # print(f'Adversarial Test Accuracy (eps={epsilon}): {acc} \t Advesarial Test Loss: {loss}')
    
    loss, acc = evaluate(model, test_loader, adv=False)
    print(f'Test Accuracy: {acc} \t Test Loss: {loss}')

# python adversarial/adv_eval.py --net=rn18 --dataset=cifar100 --expt_name=C4.3_lr0.8