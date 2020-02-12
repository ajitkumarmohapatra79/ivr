# Attention is all you need: A Pytorch Implementation

This attention network is based on the pytorch Transformer code from https://github.com/jadore801120/attention-is-all-you-need-pytorch

The original Transformer model in "[Attention is All You Need](https://arxiv.org/abs/1706.03762)" (Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, arxiv, 2017). 


The original Transformer was developed to do translation. In the original form, you have an encoding stage followed by a decoding stage. Both use multiple layers of of a Encoder/FFN structure, where the Encoder/Decoder structure further consists of positional encoding and multi-head attention network.

It is a novel sequence to sequence framework utilizes the **self-attention mechanism**, instead of Convolution operation or Recurrent structure, and achieve the state-of-the-art performance on **WMT 2014 English-to-German translation task**. (2017/06/12)

> The official Tensorflow Implementation can be found in: [tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py).

> To learn more about self-attention mechanism, you could read "[A Structured Self-attentive Sentence Embedding](https://arxiv.org/abs/1703.03130)".

<p align="center">
<img src="http://imgur.com/1krF2R6.png" width="250">
</p>


This IVR prediction model requires a simpler Attention model which eliminates the positional encoding and reduce the attention model to just a single head. The justifications are:

1. The input data already contains the order date, which gives the relative 'position' of an input vector relative to the entire order journey.

2. The multi-head attention design was needed for NLP application where a single word can have multiple meaning depending on its context. The multi-head mechanism allows the network to better capture the true semantics in the representation by projecting the first conversion into different subspaces (representing different perspectives in the word interpretation). As our data is numerical to start with, they do not face the same level of ambiguity that exist in text data. For that reason, we can simply reduce the design to a single head attention network.


# Requirement
- python 3.4+
- pytorch 1.3.1
- torchtext 0.4.0
- spacy 2.2.2+
- tqdm
- dill
- numpy


# Usage

Below is the modified instructions for running the IVR prediction code

### 1) Preprocess the data 
```bash
# The data preprocessing code can be found in the IVR/ folder. The input data can be found in ivr_embed.pkl
```

### 2) Train the model
```bash
python train-attention.py -data_cat_pkl cat_df.pkl -data_num_pkl num_df.pkl -log reduced_embed_sequence -label_smoothing -save_model trained -b 128 -warmup 128000 -epoch 1 -no_cuda
```

### 3) Test the model
```bash
python translate.py -data_pkl m30k_deen_shr.pkl -model trained.chkpt -output prediction.txt
```

```bash
python preprocess.py -raw_dir /tmp/raw_deen -data_dir ./bpe_deen -save_data bpe_vocab.pkl -codes codes.txt -prefix deen
```
