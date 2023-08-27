import torch
import torch.nn as nn
from .attention import Multihead_Attention
from .embedding import embedding
from .feed_forward import PointwiseFeedForward
from .pos_enc import pos_encoding

class encoderblock(nn.Module):
    def __init__(self, emb_dim, heads, drop_prob, pffn_dim):
        super().__init__()
# this goes as layer-norm, attention head, layer norm, feed-forward
        self.emb_dim = emb_dim
        self.heads = heads
        self.drop_prob = drop_prob
        self.pffn_dim = pffn_dim

        self.attn = Multihead_Attention(emb_dim, heads, drop_prob)
        self.feedforward = PointwiseFeedForward(emb_dim, pffn_dim, drop_prob)
        self.normlayer = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(p = drop_prob)

    def forward(self, x, mask):
        x = x + self.layer1(x, mask)
        x = x + self.layer2(x)
        return x 

    def layer1(self, x, mask): #why??
        print('before normlayer in layer 1 encode shape is : {}'.format(x.shape))
        x = self.normlayer(x)
        print('after normlayer in layer 1 encode shape is : {}'.format(x.shape))
        x = self.attn(x, x, mask)
        print('after attention in layer 1 encode shape is : {}'.format(x.shape))

        return x

    def layer2(self, x):
        x = self.normlayer(x)
        x = self.feedforward(x) #why??
        return x

class Encoder(nn.Module):
    def __init__(self, num_blocks, emb_dim, heads, drop_prob, pffn_dim):
        super().__init__()
        self.num_blocks = num_blocks
        self.emb_dim = emb_dim
        self.heads = heads
        self.drop_prob = drop_prob
        self.pffn_dim = pffn_dim

# this section goes like encoder blocks, layer-norm
        self.blocks = nn.ModuleList([encoderblock(emb_dim, heads, drop_prob, pffn_dim) for _ in range(self.num_blocks)])
        
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, x, mask):
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.layer_norm(x)
        return x


