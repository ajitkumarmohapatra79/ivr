''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from transformer.Layers_Attn import EncoderLayer


__author__ = "Yu-Hsiang Huang"


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, d_word_vec, n_layers, dropout=0.1):

        super().__init__()

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_word_vec, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_word_vec, eps=1e-6)

    def forward(self, src_seq, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        
#         enc_output = src_seq # place entity embedding right here

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(src_seq)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        enc_output = self.layer_norm(enc_output)

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, d_sentence, d_word_vec=512, n_layers=6, dropout=0.1):

        super().__init__()
        
        n_output = 2     # binary classifier

        self.encoder = Encoder(
            d_word_vec=d_word_vec, n_layers=n_layers, dropout=dropout)

        self.bin_class_prj = nn.Linear(d_sentence * d_word_vec, n_output, bias=False)  # replace input dim with total_input_dim

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

#         assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

#         self.x_logit_scale = 1.
#         if trg_emb_prj_weight_sharing:
#             # Share the weight between target word embedding & last dense layer
#             self.trg_word_prj.weight = self.decoder.trg_word_emb.weight
#             self.x_logit_scale = (d_model ** -0.5)

#         if emb_src_trg_weight_sharing:
#             self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight


    def forward(self, src_seq):

#         src_mask = get_pad_mask(src_seq, self.src_pad_idx)   # there is no idx. self.src_pad_idx is the blank order pattern
#         trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        enc_output, *_ = self.encoder(src_seq)
#         dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        # this uses only a single Linear layer. other classifier uses multiple (e.g.3) Linear layers
        flattened_output = enc_output.view(src_seq.shape[0], -1)
        seq_logit = self.bin_class_prj(flattened_output) #* self.x_logit_scale
#         print('seq_logit shape={}'.format(seq_logit.shape))

        return seq_logit#.view(-1, seq_logit.size(2))  # flatten to 2 dim (binary classification)

