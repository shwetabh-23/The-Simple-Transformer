import torch
import torch.nn as nn
from .attention import Multihead_Attention
from .feed_forward import PointwiseFeedForward

class DecoderBlock(nn.Module):
    def __init__(self, emb_dim, pffn_dim, heads, drop_prob):
        super().__init__()

        self.emb_dim = emb_dim
        self.heads  = heads
        self.pffn_dim = pffn_dim
        self.drop_prob = drop_prob

        self.attn = Multihead_Attention(emb_dim=emb_dim, heads= heads, drop_prob= drop_prob)
        self.pffn = PointwiseFeedForward(emb_dim=emb_dim, pffn_dim= pffn_dim, drop_prob= drop_prob)
        self.layernorm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(p = drop_prob)

    def layer1(self, y, ymask):
       y = self.layernorm(y)
       y = self.attn(y, y, ymask)

       return y

    def layer2(self, x, y, xmask):

        y = self.layernorm(y)
        y = self.attn(x, y, xmask)

        return y

    def layer3(self, y):

        y = self.layernorm(y)
        y = self.pffn(y)

        return y

    def forward(self, y, ymask, x, xmask):
        y = y + self.layer1(y, ymask)
        y = y + self.layer2(x, y, xmask)
        y = y + self.layer3(y)

        return y

class Decoder(nn.Module):
    def __init__(self, num_blocks, emb_dim, heads, drop_prob, pffn_dim):
        super().__init__()

        self.blocks = nn.ModuleList([DecoderBlock(emb_dim, pffn_dim, heads, drop_prob) for _ in range(num_blocks)])
        self.layernorm = nn.LayerNorm(emb_dim)

    def forward(self, x, y, xmask, ymask):
        
        for block in self.blocks:
            y = block(y, ymask, x, xmask)
        y = self.layernorm(y)

        return y

        

