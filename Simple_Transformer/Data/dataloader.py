"""
dataloader  
"""
from .dataset import create_dataset, load_dataset
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from .vocab import EOS_IDX, SOS_IDX, PAD_IDX, UNK_IDX, Vocab, load_vocab_pair

def create_masks(source, target):
    """mask creating function"""

    source_pad_mask = (source != PAD_IDX).unsqueeze(1)
    targe_pad_mask = (target != PAD_IDX).unsqueeze(1)

    max_length = target.shape[1]
    attn_mat = (max_length, max_length)

    full_mask = torch.full(attn_mat, 1)
    triangular_mat = torch.tril(full_mask)
    tri_mat = triangular_mat.unsqueeze(0)

    return source_pad_mask, targe_pad_mask & tri_mat



def make_data_loader(dataset, source_vocab, target_vocab, batch_size, device):
    """data loader function with an explcotit collate"""

    def collate_fn(batch):

        source_tokens = []
        target_tokens = []

        max_length = 0
        max_len_source = 0
        max_len_target = 0

        for i, (source_sent, target_sent) in enumerate(batch):
            
            for i in range(len(source_vocab(source_sent))):
                max_len_source = max(max_len_source, len(source_vocab(source_sent)))

            for i in range(len(target_vocab(target_sent))):
                max_len_target = max(max_len_target, len(target_vocab(target_sent)))
            
            
        max_len_target += 1
        max_length = max(max_len_source, max_len_target)
            
        #     if max(len(source_sent.split()), len(target_sent.split())) > max_length:
        #         max_length = max(len(source_sent.split()), len(target_sent.split()))
        # max_length += 2

        for i, (source_sent, target_sent) in enumerate(batch):
            source_inp = source_vocab(source_sent)
            if len(source_inp) < max_length:
                source_inp.extend([PAD_IDX] * (max_length - len(source_inp)))

            target_inp = target_vocab(target_sent)
            target_inp = [SOS_IDX] + target_inp + [EOS_IDX]

            if len(target_inp) < max_length:
                target_inp.extend([PAD_IDX] * (max_length - len(target_inp) + 1))

            source_tokens.append(Tensor(source_inp))
            target_tokens.append(Tensor(target_inp))
        breakpoint()
        source = pad_sequence(source_tokens, True, PAD_IDX)
        target = pad_sequence(target_tokens, True, PAD_IDX)

        labels = target[:, 1:]
        target = target[:, :-1]

        source_masks, target_masks = create_masks(source, target)
        
        return [source.to(device), target.to(device), labels.to(device), source_masks.to(device), target_masks.to(device)]
    
    return DataLoader(dataset=dataset, batch_size= batch_size, collate_fn= collate_fn)