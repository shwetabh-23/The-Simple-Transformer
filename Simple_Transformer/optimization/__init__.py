from .loss import TransationalLoss
from .Scheduler import Scheduler
import torch
import torch.nn as nn

def make_optimizer(parameters, name, **kwargs):
    optim_class = eval(name)
    optim = optim_class(parameters, **kwargs)
    return optim

def make_scheduler(Optimizer, name, **kwargs):
    sch_class = eval(name)
    sch = sch_class(Optimizer, **kwargs)
    return sch

def make_loss_func(name, **kwargs):
    loss_func_class = eval(name)
    loss_func = loss_func_class(**kwargs)
    return loss_func
