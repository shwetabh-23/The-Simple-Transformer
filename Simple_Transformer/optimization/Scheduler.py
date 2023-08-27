def calc_lr(emb_dim, step, warm_steps):
    return emb_dim ** (-0.5) * min(step ** (-0.5), warm_steps ** (-1.5))

import torch
import torch.nn as nn
from torch.optim import Optimizer
from  torch.optim.lr_scheduler import LRScheduler

class Scheduler(LRScheduler):
    def __init__(self, Optimizer : Optimizer, emb_dim, warm_steps, last_epoch = -1, verbose = False):
        
        self.emb_dim = emb_dim
        self.warm_steps = warm_steps
        self.num_param_groups = len(Optimizer.param_groups)

        super().__init__(Optimizer, last_epoch, verbose)

    def get_lr(self):
        lr = calc_lr(self._step_count, self.emb_dim, self.warm_steps)
        return [lr] * self.num_param_groups

