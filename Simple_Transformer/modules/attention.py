import torch
import torch.nn as nn
import math
from torch.nn import functional as F

class Multihead_Attention(nn.Module):
    def __init__(self, emb_dim, heads, drop_prob):
        super().__init__()

        assert emb_dim % heads == 0

        self.emb_dim = emb_dim
        self.heads = heads
        self.head_dim = emb_dim // heads

        self.query = nn.Linear(emb_dim, emb_dim)
        self.key = nn.Linear(emb_dim, emb_dim)
        self.value = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(p = drop_prob)
        self.output = nn.Linear(emb_dim, emb_dim)

    def forward(self, x, y, mask):
        query = self.query(x)
        key = self.key(y)
        value = self.value(y)
# the input shape now should be (batch size, number of words in the sentence, embedding dimentions). We need to reshape this to split the last dimension into ->
# number of dimension heads *  dimension heads.
        batch_size = x.size(0)
        query = query.view(batch_size, -1, self.heads, self.head_dim)
        key = key.view(batch_size, -1, self.heads, self.head_dim)
        value = value.view(batch_size, -1, self.heads, self.head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        attn = Attention(query, key, value, mask)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.emb_dim)

        out = self.output(attn)
        out = self.dropout(out)

        return out

def Attention(query, key, value, mask):
        print('attention Starting : shape of 1st and 2nd matrix : {} and {}'.format(query.shape, (key.transpose(-2, -1)).shape))
        score = torch.matmul(query, key.transpose(-2, -1))
        score = score/math.sqrt(query.shape[-1])
        print('the shape of scores and mask are : {} and {}'.format(score.shape, mask.shape))
        
        if mask is not None :
            score.masked_fill(mask == 0, -1e-9) 
        weights = F.softmax(score, dim = -1)
        return torch.matmul(weights, value)


