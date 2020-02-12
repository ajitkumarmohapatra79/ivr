''' Define the Layers '''
import torch.nn as nn
import torch
from transformer.SubLayers_Attn import SingleAttention


__author__ = "Yu-Hsiang Huang"


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = SingleAttention(d_model, dropout=dropout)
#         self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input)
#         enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn
