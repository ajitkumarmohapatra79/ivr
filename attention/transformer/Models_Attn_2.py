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
            self, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1, n_position=200):

        super().__init__()

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        
        enc_output = src_seq # place entity embedding right here

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        enc_output = self.layer_norm(enc_output)

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


# +
class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, 
            d_word_vec=512, d_model=512, d_inner=2048,            # what d_model do we want? total_input_dim is 1252
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True):

        super().__init__()
        
        self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_size])
        self.embedding_dropout = nn.Dropout(p)
        self.batch_norm_num = nn.BatchNorm1d(num_numerical_cols)

        all_layers = []
        num_categorical_cols = sum((nf for ni, nf in embedding_size))
        input_size = num_categorical_cols + num_numerical_cols
        
        n_output = 2     # binary classifier

        self.encoder = Encoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.bin_class_prj = nn.Linear(d_model, n_output, bias=False)  # replace input dim with total_input_dim

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


    def forward(self, x_categorical, x_numerical):
        embeddings = []
        for i,e in enumerate(self.all_embeddings):
            embeddings.append(e(x_categorical[:,i]))
        x = torch.cat(embeddings, 1)
        x = self.embedding_dropout(x)

        x_numerical = self.batch_norm_num(x_numerical)
        x = torch.cat([x, x_numerical], 1)
#         src_mask = get_pad_mask(src_seq, self.src_pad_idx)   # there is no idx. self.src_pad_idx is the blank order pattern
#         trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        enc_output, *_ = self.encoder(x)
#         dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        # this uses only a single Linear layer. other classifier uses multiple (e.g.3) Linear layers
        seq_logit = self.bin_class_prj(enc_output) #* self.x_logit_scale

#         return seq_logit.view(-1, seq_logit.size(2))  # why size(2)? batch, sentence length, idx(dim=1)
        return seq_logit
