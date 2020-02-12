

import torch
import pickle as pkl

from torch.utils.data import Dataset, DataLoader

class IVRDataset(Dataset):

    def __init__(self, csv_path, sentence_dim, word_dim):
#         self.data = pd.read_csv(csv_path)
        with open(csv_path,'rb') as reduced_embed_sequence_pkl:
            self.data =  pkl.load(reduced_embed_sequence_pkl)
        self.sentence_dim = sentence_dim
        self.word_dim = word_dim
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        # first the embeding input feature
        embedded_tensor = torch.as_tensor(self.data.iloc[index]['embedded'], dtype=torch.float32)  # one sentence
#         batched_tensor = np.empty((0, self.sentence_dim, self.word_dim), float)
        # print('input_x shape={}'.format(input_x.shape))
#         for embedded_tensor in embedded_tensors:
#             expanded_tensor = np.expand_dims(embedded_tensor, axis=0)
#         #     print('expanded_tensor shape={}'.format(expanded_tensor.shape))
#             batched_tensor = np.vstack((batched_tensor, expanded_tensor))
#         print('embedded_tensor shape={}'.format(embedded_tensor.shape))
#         print('type embedded_tensor={}'.format(type(embedded_tensor)))
        # next the label
        label_tensor = torch.as_tensor(self.data.iloc[index]['label'], dtype=torch.float32)
#         print('label_tensor shape={}'.format(label_tensor.shape))

        return embedded_tensor, label_tensor 

def prepare_dataloaders(opt, device):
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
    trainset = IVRDataset(opt.data_pkl, opt.max_token_seq_len, word_dim)
    total_samples = trainset.__len__()
    print('total_samples={}'.format(total_samples))    
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
#     print('train item 0 src={}'.format(train.__getitem__(0).src))
#     train_iterator = BucketIterator(train, batch_size=batch_size, device=device, train=True)
#     val_iterator = BucketIterator(val, batch_size=batch_size, device=device)

    return train_loader
