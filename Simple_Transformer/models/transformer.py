import torch
import torch.nn as nn
from ..modules import Multihead_Attention, embedding, PointwiseFeedForward, pos_encoding, Encoder, Decoder

class Transformer(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, emb_dim, max_positions, drop_prob, num_blocks, heads, pffn_dim):
        super().__init__()

        self.input_embedding = embedding(input_vocab_size, emb_dim)
        self.output_embedding = embedding(output_vocab_size, emb_dim)
        self.pos_enc = pos_encoding(max_positions, emb_dim, drop_prob)

        self.encoder = Encoder(num_blocks, emb_dim, heads, drop_prob, pffn_dim)
        self.decoder = Decoder(num_blocks, emb_dim, heads, drop_prob, pffn_dim)
        self.projection = nn.Linear(emb_dim, output_vocab_size)

    def encode(self, x, xmask):
        
        x = self.input_embedding(x)
        x = self.pos_enc(x)
        x = self.encoder(x = x, mask = xmask)
        return x

    def decode(self, y, ymask, x, xmask):

        y = self.output_embedding(y)
        y = self.pos_enc(y)
        y = self.decoder(x, y, xmask, ymask)
        return self.projection(y)

    def forward(self, x, xmask, y, ymask):

        x = self.encode(x, xmask)
        y = self.decode(y, ymask, x, xmask)
        return y





