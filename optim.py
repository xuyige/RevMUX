
from torch.optim import Optimizer
from typing import List


class DoubleOptimizer:

    def __init__(self, optimizer_list: List[Optimizer]):
        self.optimizer_list = optimizer_list
        self.param_groups = []
        self.total_optim_param = 0
        for optimizer in self.optimizer_list:
            for param_groups in optimizer.param_groups:
                self.param_groups = self.param_groups + list(param_groups['params'])
                for param in param_groups['params']:
                    self.total_optim_param += param.numel()

        self.param_groups = [{'params': self.param_groups}]

    def step(self, *args, **kwargs):
        for optimizer in self.optimizer_list:
            optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs):
        for optimizer in self.optimizer_list:
            optimizer.zero_grad(*args, **kwargs)
