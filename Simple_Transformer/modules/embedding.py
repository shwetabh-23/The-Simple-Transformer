import torch
import torch.nn as nn
import math

class embedding(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)

    def forward(self, x):
        x = self.embedding(x.to(torch.long))
        x = x * math.sqrt(self.emb_dim)
        return x
        