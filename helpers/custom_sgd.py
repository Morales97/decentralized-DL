import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, required, _use_grad_for_differentiable
from typing import List, Optional

import pdb
import sys
import os
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from helpers.custom_optim import CustomOptimizer

__all__ = ['SGD', 'sgd']

class CustomSGD(CustomOptimizer):
    
    def __init__(self, params_x, params_y, params_v, lr=required, alpha=0, beta=0, variant=0, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, *, maximize=False, foreach: Optional[bool] = None,
                 differentiable=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        maximize=maximize, foreach=foreach,
                        differentiable=differentiable)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        self.alpha=alpha
        self.beta=beta
        self.variant=variant
        super(CustomSGD, self).__init__(params_x, params_y, params_v, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('differentiable', False)

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # DM: Implementing the following algorithm
        # v_{t+1} = v_{t} - lr·g(y_t)
        # x_{t+1} = (1-a)·x_t + a·v_{t+1}
        # y_{t+1} = (1-b)·x_{t+1} + b·v_{t+1}

        # v_{t+1} = v_{t} - lr·g(y_t)
        for group_y, group_v in zip(self.param_groups, self.param_groups_v):
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            has_sparse_grad = False

            for p_y, p_v in zip(group_y['params'], group_v['params']):
                if p_y.grad is not None:
                    params_with_grad.append(p_v)
                    d_p_list.append(p_y.grad)
                    if p_y.grad.is_sparse:
                        has_sparse_grad = True

                    state = self.state[p_y]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            
            sgd(params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=group_y['weight_decay'],
                momentum=group_y['momentum'],
                lr=group_y['lr'],
                dampening=group_y['dampening'],
                nesterov=group_y['nesterov'],
                maximize=group_y['maximize'], 
                has_sparse_grad=has_sparse_grad,
                foreach=group_y['foreach'])

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        # x_{t+1} = (1-a)·x_t + a·v_{t+1}
        if self.variant == 0:
            for group_x, group_v in zip(self.param_groups_x, self.param_groups_v):
                for p_x, p_v in zip(group_x['params'], group_v['params']):
                    _combine_params(p_x, p_v, self.alpha)

        # x_{t+1} = ·x_t + a·v_{t+1}
        elif self.variant == 1:
            for group_x, group_v in zip(self.param_groups_x, self.param_groups_v):
                for p_x, p_v in zip(group_x['params'], group_v['params']):
                    for p1, p2 in zip(p_x, p_v):
                        p1.add_(p2, alpha=self.alpha)

        # y_{t+1} = (1-b)·x_{t+1} + b·v_{t+1}
        for group_x, group_y, group_v in zip(self.param_groups_x, self.param_groups, self.param_groups_v):
            for p_x, p_y, p_v in zip(group_x['params'], group_y['params'], group_v['params']):
                _combine_params_2(p_y, p_x, p_v, self.beta)


        return loss



def sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
        # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
        has_sparse_grad: bool = None,
        foreach: bool = None,
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        maximize: bool):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    if foreach and not torch.jit.is_scripting():
        func = None # _multi_tensor_sgd
    else:
        func = _single_tensor_sgd

    func(params,
         d_p_list,
         momentum_buffer_list,
         weight_decay=weight_decay,
         momentum=momentum,
         lr=lr,
         dampening=dampening,
         nesterov=nesterov,
         has_sparse_grad=has_sparse_grad,
         maximize=maximize)

def _single_tensor_sgd(params: List[Tensor],
                       d_p_list: List[Tensor],
                       momentum_buffer_list: List[Optional[Tensor]],
                       *,
                       weight_decay: float,
                       momentum: float,
                       lr: float,
                       dampening: float,
                       nesterov: bool,
                       maximize: bool,
                       has_sparse_grad: bool):

    for i, param in enumerate(params):
        d_p = d_p_list[i] if not maximize else -d_p_list[i]

        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        param.add_(d_p, alpha=-lr)

# DM: for example, x_{t+1} = (1-a)·x_t + a·v_{t+1}
def _combine_params(params_1: List[Tensor],
                       params_2: List[Tensor],
                       alpha: float):

    for p1, p2 in zip(params_1, params_2):
        p1.mul_(1-alpha)
        p1.add_(p2, alpha=alpha)

# DM: for example, y_{t+1} = (1-b)·x_{t+1} + b·v_{t+1}
def _combine_params_2(params_1: List[Tensor],
                       params_2: List[Tensor],
                       params_3: List[Tensor],
                       alpha: float):

    for p1, p2, p3 in zip(params_1, params_2, params_3):
        p1.zero_()
        p1.add_(p2, alpha=(1-alpha))
        p1.add_(p3, alpha=alpha)
