'''
This script handles the training process.
'''

# +
import argparse
import math
import time
import dill as pickle
from tqdm import tqdm

import pandas as pd
import pickle as pkl
import numpy as np

# +
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchtext.data import BucketIterator
# from torch.utils.data import TensorDataset

from torch.utils.data import DataLoader, random_split
# -

import transformer.Constants as Constants
from transformer.Models_Attn import Transformer
from transformer.Optim import ScheduledOptim


# +
# def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
#     ''' Apply label smoothing if needed '''

#     loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

#     pred = pred.max(1)[1]
#     gold = gold.contiguous().view(-1)
#     non_pad_mask = gold.ne(trg_pad_idx)
#     n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
#     n_word = non_pad_mask.sum().item()

#     return loss, n_correct, n_word

# +
# def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
#     ''' Calculate cross entropy loss, apply label smoothing if needed. '''

#     gold = gold.contiguous().view(-1)

#     if smoothing:
#         eps = 0.1
#         n_class = pred.size(1)

#         one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
#         one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
#         log_prb = F.log_softmax(pred, dim=1)

#         non_pad_mask = gold.ne(trg_pad_idx)
#         loss = -(one_hot * log_prb).sum(dim=1)
#         loss = loss.masked_select(non_pad_mask).sum()  # average later
#     else:
#         loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
#     return loss

# +
# def patch_src(src, pad_idx):
#     src = src.transpose(0, 1)
#     return src

# +
# def patch_trg(trg, pad_idx):
#     trg = trg.transpose(0, 1)
#     trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
#     return trg, gold

# +
# def train_epoch(model, training_data, optimizer, opt, device, smoothing):
#     ''' Epoch operation in training phase'''

#     model.train()
#     total_loss, n_word_total, n_word_correct = 0, 0, 0 
    
#     item = next(x for x in training_data)
#     print('type item={}'.format(item))
# #     print('item src={}'.format(item.src))

#     desc = '  - (Training)   '
#     # replace this call 
#     # define a Batch which provides a tensor of call records
#     for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):

#         # prepare data
# #         print('batch.src={}'.format(batch.src))
#         src_seq = patch_src(batch.src, opt.src_pad_idx).to(device)
# #         print('src_seq={}'.format(src_seq))
#         trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, opt.trg_pad_idx))

#         # forward
#         optimizer.zero_grad()
#         pred = model(src_seq, trg_seq)

#         # backward and update parameters
#         loss, n_correct, n_word = cal_performance(
#             pred, gold, opt.trg_pad_idx, smoothing=smoothing) 
#         loss.backward()
#         optimizer.step_and_update_lr()

#         # note keeping
#         n_word_total += n_word
#         n_word_correct += n_correct
#         total_loss += loss.item()

#     loss_per_word = total_loss/n_word_total
#     accuracy = n_word_correct/n_word_total
#     return loss_per_word, accuracy

# +
# def eval_epoch(model, validation_data, device, opt):
#     ''' Epoch operation in evaluation phase '''

#     model.eval()
#     total_loss, n_word_total, n_word_correct = 0, 0, 0

#     desc = '  - (Validation) '
#     with torch.no_grad():
#         for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):

#             # prepare data
#             src_seq = patch_src(batch.src, opt.src_pad_idx).to(device)
#             trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, opt.trg_pad_idx))

#             # forward
#             pred = model(src_seq, trg_seq)
#             loss, n_correct, n_word = cal_performance(
#                 pred, gold, opt.trg_pad_idx, smoothing=False)

#             # note keeping
#             n_word_total += n_word
#             n_word_correct += n_correct
#             total_loss += loss.item()

#     loss_per_word = total_loss/n_word_total
#     accuracy = n_word_correct/n_word_total
#     return loss_per_word, accuracy

# +
def train(model, training_data, optimizer, device, opt):
    ''' Start training '''

#     log_train_file, log_valid_file = None, None

#     if opt.log:
#         log_train_file = opt.log + '.train.log'
#         log_valid_file = opt.log + '.valid.log'

#         print('[Info] Training performance will be written to file: {} and {}'.format(
#             log_train_file, log_valid_file))

#         with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
#             log_tf.write('epoch,loss,ppl,accuracy\n')
#             log_vf.write('epoch,loss,ppl,accuracy\n')

#     def print_performances(header, loss, accu, start_time):
#         print('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
#               'elapse: {elapse:3.3f} min'.format(
#                   header=f"({header})", ppl=math.exp(min(loss, 100)),
#                   accu=100*accu, elapse=(time.time()-start_time)/60))

    #valid_accus = []
#     valid_losses = []
    
    for i in range(opt.epoch):
        i += 1
        for i, batch in enumerate(training_data):
            y_pred = model(training_data)
            single_loss = loss_function(y_pred, train_outputs)
            aggregated_losses.append(single_loss)

            if i%25 == 1:
                print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

            optimizer.zero_grad()
            single_loss.backward()
    #         optimizer.step()
            optimizer.step_and_update_lr()
    
    
#     for epoch_i in range(opt.epoch):
#         print('[ Epoch', epoch_i, ']')

#         start = time.time()
#         train_loss, train_accu = train_epoch(
#             model, training_data, optimizer, opt, device, smoothing=opt.label_smoothing)
#         print_performances('Training', train_loss, train_accu, start)

#         start = time.time()
#         valid_loss, valid_accu = eval_epoch(model, validation_data, device, opt)
#         print_performances('Validation', valid_loss, valid_accu, start)

#         valid_losses += [valid_loss]

#         checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': model.state_dict()}

#         if opt.save_model:
#             if opt.save_mode == 'all':
#                 model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
#                 torch.save(checkpoint, model_name)
#             elif opt.save_mode == 'best':
#                 model_name = opt.save_model + '.chkpt'
#                 if valid_loss <= min(valid_losses):
#                     torch.save(checkpoint, model_name)
#                     print('    - [Info] The checkpoint file has been updated.')

#         if log_train_file and log_valid_file:
#             with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
#                 log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
#                     epoch=epoch_i, loss=train_loss,
#                     ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
#                 log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
#                     epoch=epoch_i, loss=valid_loss,
#                     ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))
# -

def main():
    ''' 
    Usage:
    python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 256 -warmup 128000
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument('-data_cat_pkl', default=None)     # all-in-1 data pickle or bpe field
    parser.add_argument('-data_num_pkl', default=None)     # all-in-1 data pickle or bpe field

    parser.add_argument('-train_path', default=None)   # bpe encoded data
    parser.add_argument('-val_path', default=None)     # bpe encoded data

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=2048)

    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-warmup','--n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    if not opt.log and not opt.save_model:
        print('No experiment result will be saved.')
        raise

    if opt.batch_size < 2048 and opt.n_warmup_steps <= 4000:
        print('[Warning] The warmup steps may be not enough.\n'\
              '(sz_b, warmup) = (2048, 4000) is the official setting.\n'\
              'Using smaller batch w/o longer warmup may cause '\
              'the warmup stage ends with only little data trained.')

    device = torch.device('cuda' if opt.cuda else 'cpu')

    #========= Loading Dataset =========#
    # training_data and validation_data should be BucketIterator of TensorDataSets
    training_cat_data, validation_cat_data = prepare_cat_dataloaders(opt, device)
    training_num_data, validation_num_data = prepare_num_dataloaders(opt, device)

    print(opt)

    transformer = Transformer(
#         opt.src_vocab_size,
#         opt.trg_vocab_size,
#         src_pad_idx=opt.src_pad_idx,
#         trg_pad_idx=opt.trg_pad_idx,
        trg_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_trg_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001)
#     optimizer = ScheduledOptim(
#         optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
#         2.0, opt.d_model, opt.n_warmup_steps)
    import time
    start_time = time.time()
    train(transformer, training_data, optimizer, device, opt)
        
    print("--- %s seconds ---" % (time.time() - start_time))



# +
# import torch
# from torch.utils import data

# class IVRDataset(data.Dataset):

#     def __init__(self, csv_path, chunkSize, sentence_dim, word_dim):
#         self.chunksize = chunkSize
#         self.reader = pd.read_csv(csv_path, sep=',', chunksize=self.chunksize, header=None, iterator=True)
#         self.sentence_dim = sentence_dim
#         self.word_dim = word_dim
    
#     def __len__(self):
#         return self.chunksize

#     def __getitem__(self, index):
#         data = self.reader.get_chunk(self.chunksize)
#         # first the embeding input feature
#         embedded_tensors = data['embedded'].values
#         batched_tensor = np.empty((0, self.sentence_dim, self.word_dim), float)
#         # print('input_x shape={}'.format(input_x.shape))
#         for embedded_tensor in embedded_tensors:
#             expanded_tensor = np.expand_dims(embedded_tensor, axis=0)
#         #     print('expanded_tensor shape={}'.format(expanded_tensor.shape))
#             batched_tensor = np.vstack((batched_tensor, expanded_tensor))
#         print('batched_tensor shape={}'.format(batched_tensor.shape))

#         # next the label
#         label_tensor = torch.as_tensor(data['label'].values, dtype=torch.float32)
#         print('label_tensor shape={}'.format(label_tensor.shape))

#         return batched_tensor, label_tensor 

# +
import torch
from torch.utils import data

class IVR_cat_Dataset(data.Dataset):

    def __init__(self, cat_csv_path, sentence_dim, word_dim):
#         self.data = pd.read_csv(csv_path)
        with open(cat_csv_path,'rb') as cat_data_pkl:
            self.data_df =  pkl.load(cat_data_pkl)
        self.sentence_dim = sentence_dim
        self.word_dim = word_dim
    
    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, index):
        cat_columns = self.data_df.columns.to_list()
        code_values = []
        for column in cat_columns:
            cat_values = dataset[column].cat.codes.values
            code_values.append(cat_values)
        categorical_data = np.stack(code_values, 1)
        categorical_data = torch.tensor(categorical_data, dtype=torch.int64)
        
        # first the embeding input feature
        embedded_tensors = self.data['embedded'].values
        batched_tensor = np.empty((0, self.sentence_dim, self.word_dim), float)
        # print('input_x shape={}'.format(input_x.shape))
        for embedded_tensor in embedded_tensors:
            expanded_tensor = np.expand_dims(embedded_tensor, axis=0)
        #     print('expanded_tensor shape={}'.format(expanded_tensor.shape))
            batched_tensor = np.vstack((batched_tensor, expanded_tensor))
        print('batched_tensor shape={}'.format(batched_tensor.shape))

        # next the label
        label_tensor = torch.as_tensor(self.data['label'].values, dtype=torch.float32)
        print('label_tensor shape={}'.format(label_tensor.shape))

        return batched_tensor, label_tensor 


# -

def prepare_cat_dataloaders(opt, device):
    batch_size = opt.batch_size
    word_dim = 263
#     data = pickle.load(open(opt.data_pkl, 'rb'))   # need to generate a pkl'ed dict 
                                                    # want 'setting', 'train', and 'valid'
                                                    # settings has max_len, train and valid are TensorDataset

#     opt.max_token_seq_len = data['settings'].max_len      # max length = max orders per account, currently set to 50 
    opt.max_token_seq_len = 50      # max length = max orders per account, currently set to 50 
#     opt.src_pad_idx = data['vocab']['src'].vocab.stoi[Constants.PAD_WORD]
#     opt.trg_pad_idx = data['vocab']['trg'].vocab.stoi[Constants.PAD_WORD]

#     opt.src_vocab_size = len(data['vocab']['src'].vocab)
#     opt.trg_vocab_size = len(data['vocab']['trg'].vocab)

    #========= Preparing Model =========#
#     if opt.embs_share_weight:
#         assert data['vocab']['src'].vocab.stoi == data['vocab']['trg'].vocab.stoi, \
#             'To sharing word embedding the src/trg word2idx table shall be the same.'
    dataset = IVRDataset(opt.data_pkl, opt.max_token_seq_len, word_dim)
    total_samples = dataset.__len__()
    print('total_samples={}'.format(total_samples))
    trainset, valset = random_split(dataset, [int(total_samples*0.8), int(total_samples*0.2)])
    
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=0)

#     print('train item 0 src={}'.format(train.__getitem__(0).src))
#     train_iterator = BucketIterator(train, batch_size=batch_size, device=device, train=True)
#     val_iterator = BucketIterator(val, batch_size=batch_size, device=device)

    return train_loader, val_loader


def prepare_num_dataloaders(opt, device):
    batch_size = opt.batch_size
    word_dim = 263
#     data = pickle.load(open(opt.data_pkl, 'rb'))   # need to generate a pkl'ed dict 
                                                    # want 'setting', 'train', and 'valid'
                                                    # settings has max_len, train and valid are TensorDataset

#     opt.max_token_seq_len = data['settings'].max_len      # max length = max orders per account, currently set to 50 
    opt.max_token_seq_len = 50      # max length = max orders per account, currently set to 50 
#     opt.src_pad_idx = data['vocab']['src'].vocab.stoi[Constants.PAD_WORD]
#     opt.trg_pad_idx = data['vocab']['trg'].vocab.stoi[Constants.PAD_WORD]

#     opt.src_vocab_size = len(data['vocab']['src'].vocab)
#     opt.trg_vocab_size = len(data['vocab']['trg'].vocab)

    #========= Preparing Model =========#
#     if opt.embs_share_weight:
#         assert data['vocab']['src'].vocab.stoi == data['vocab']['trg'].vocab.stoi, \
#             'To sharing word embedding the src/trg word2idx table shall be the same.'
    dataset = IVRDataset(opt.data_pkl, opt.max_token_seq_len, word_dim)
    total_samples = dataset.__len__()
    print('total_samples={}'.format(total_samples))
    trainset, valset = random_split(dataset, [int(total_samples*0.8), int(total_samples*0.2)])
    
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=0)

#     print('train item 0 src={}'.format(train.__getitem__(0).src))
#     train_iterator = BucketIterator(train, batch_size=batch_size, device=device, train=True)
#     val_iterator = BucketIterator(val, batch_size=batch_size, device=device)

    return train_loader, val_loader

if __name__ == '__main__':
    main()


