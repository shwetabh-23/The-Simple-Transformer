import torch
import torch.nn as nn
from ..Data import PAD_IDX

class TransationalLoss(nn.Module):
    def __init__(self, label_smoothing = 0):
        super().__init__()

        self.lossfunc = nn.CrossEntropyLoss(ignore_index= PAD_IDX, label_smoothing = label_smoothing)

    def forward(self, logits, target):
        req_shape = logits.shape[-1]
        logits = logits.reshape(-1, req_shape)
        target = target.reshape(-1).long()

        return (self.lossfunc(logits, target))
