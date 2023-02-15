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

class SGD_GradStats(Optimizer):
    '''
    Same as default SGD, but keep track of 1st and 2nd moments of gradient 
    '''
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, *, maximize: bool = False, foreach: Optional[bool] = None,
                 differentiable: bool = False, beta1: float = 0.99, beta2: float = 0.99):
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
        self.beta1 = beta1
        self.beta2 = beta2
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('differentiable', False)

    def _init_group(self, group, params_with_grad, d_p_list, momentum_buffer_list, exp_avgs, exp_avg_sqs):
        has_sparse_grad = False

        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                d_p_list.append(p.grad)
                if p.grad.is_sparse:
                    has_sparse_grad = True

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state['momentum_buffer'])
                
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

        return has_sparse_grad


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

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            exp_avgs = []
            exp_avg_sqs = []

            has_sparse_grad = self._init_group(group, params_with_grad, d_p_list, momentum_buffer_list, exp_avgs, exp_avg_sqs)

            sgd(params_with_grad,
                d_p_list,
                momentum_buffer_list,
                exp_avgs,
                exp_avg_sqs,
                self.beta1,
                self.beta2,
                weight_decay=group['weight_decay'],
                momentum=group['momentum'],
                lr=group['lr'],
                dampening=group['dampening'],
                nesterov=group['nesterov'],
                maximize=group['maximize'],
                has_sparse_grad=has_sparse_grad,
                foreach=group['foreach'])

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss


def sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
        # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        beta1: float,
        beta2: float,
        has_sparse_grad: bool = None,
        foreach: Optional[bool] = None,
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

    # if foreach is None:
    #     # why must we be explicit about an if statement for torch.jit.is_scripting here?
    #     # because JIT can't handle Optionals nor fancy conditionals when scripting
    #     if not torch.jit.is_scripting():
    #         _, foreach = _default_to_fused_or_foreach([params, d_p_list, momentum_buffer_list],
    #                                                   differentiable=False, has_fused=False)
    #     else:
    #         foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    if foreach and not torch.jit.is_scripting():
        raise RuntimeError('DM: not implemented')
    else:
        func = _single_tensor_sgd

    func(params,
         d_p_list,
         momentum_buffer_list,
         exp_avgs, 
         exp_avg_sqs,
         beta1,
         beta2,
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
                       exp_avgs: List[Tensor],
                       exp_avg_sqs: List[Tensor],
                       beta1: float,
                       beta2: float,
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
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]

        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(d_p, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(d_p, d_p, value=1 - beta2)

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
