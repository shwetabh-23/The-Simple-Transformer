import torch
import torch.nn as nn

class PointwiseFeedForward(nn.Module):
    def __init__(self, emb_dim, pffn_dim, drop_prob):
        super().__init__()

        self.pffn = nn.Sequential(nn.Linear(emb_dim, pffn_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(p = drop_prob),
        nn.Linear(pffn_dim, emb_dim),
        nn.Dropout(p = drop_prob)
        )

    def forward(self, x):
        return self.pffn(x)

        
        

