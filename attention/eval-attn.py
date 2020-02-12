''' Translate input text with trained model. '''

# +
import torch
import argparse
# import dill as pickle
from tqdm import tqdm

import pickle as pkl
# -

import transformer.Constants as Constants
from torchtext.data import Dataset
from transformer.Models import Transformer
from transformer.Translator import Translator

from Data_util import IVRDataset, prepare_dataloaders


def load_model(opt, device):

    checkpoint = torch.load(opt.model, map_location=device)
    model_opt = checkpoint['settings']

    model = Transformer(
        model_opt.src_vocab_size,
        model_opt.trg_vocab_size,

        model_opt.src_pad_idx,
        model_opt.trg_pad_idx,

        trg_emb_prj_weight_sharing=model_opt.proj_share_weight,
        emb_src_trg_weight_sharing=model_opt.embs_share_weight,
        d_k=model_opt.d_k,
        d_v=model_opt.d_v,
        d_model=model_opt.d_model,
        d_word_vec=model_opt.d_word_vec,
        d_inner=model_opt.d_inner_hid,
        n_layers=model_opt.n_layers,
        n_head=model_opt.n_head,
        dropout=model_opt.dropout).to(device)

    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')
    return model 


@torch.no_grad()
def eval(model, validation_data, loss_function, optimizer, device, opt):
#     with torch.no_grad():
#         y_val = model(validation_data)
#         loss = loss_function(y_val, test_outputs)
#     print(f'Loss: {loss:.8f}')  
    
    # model, images, labels, pre_model, save_dir, sum_dir, batch_size, lr, log_after, cuda
    un_confusion_meter = tnt.meter.ConfusionMeter(10, normalized=False)
    confusion_meter = tnt.meter.ConfusionMeter(10, normalized=True)
#     model.load_state_dict(torch.load(pre_model))
#     print('log: resumed model {} successfully!'.format(pre_model))
#     _, _, test_loader  = get_dataloaders(base_folder=base_folder, batch_size=batch_size)
    net_accuracy, net_loss = [], []
    correct_count = 0
    total_count = 0
    
    aggregated_losses = []
#     print('batch size = {}'.format(batch_size))
    model.eval()  # put in eval mode first
    for idx, data in enumerate(validation_data):
        embedded_tensor = batch[0]
#             print('embedded_tensor={}'.format(embedded_tensor))
        y_pred = model(embedded_tensor)
#             print('y_pred shape={}'.format(y_pred.shape))
#             print('batch[1] shape={}'.format(batch[1].shape))
        val_outputs = batch[1].long()
#             print('train_outputs={}'.format(train_outputs))
#             print('train_outputs shape={}'.format(train_outputs.shape))
        single_loss = loss_function(y_pred, val_outputs)
        aggregated_losses.append(single_loss)
        un_confusion_meter.add(predicted=pred, target=label)
        confusion_meter.add(predicted=pred, target=label)

        ###############################
        # pred = pred.view(-1)
        # pred = pred.cpu().numpy()
        # label = label.cpu().numpy()
        # print(pred.shape, label.shape)
        ###############################

        # get accuracy metric
        # correct_count += np.sum((pred == label))
        # print(pred, label)
        # get accuracy metric
#         if 'one_hot' in kwargs.keys():
#             if kwargs['one_hot']:
#                 batch_correct = (torch.argmax(label, dim=1).eq(pred.long())).double().sum().item()
#         else:
        batch_correct = (label.eq(pred.long())).sum().item()
        # print(label.shape, pred.shape)
        # break
        correct_count += batch_correct
        # print(batch_correct)
        total_count += np.float(batch_size)
        net_loss.append(loss.item())
        if idx % log_after == 0:
            print('log: on {}'.format(idx))

        #################################
    mean_loss = np.asarray(net_loss).mean()
    mean_accuracy = correct_count * 100 / total_count
    print(correct_count, total_count)
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    print('log: test:: total loss = {:.5f}, total accuracy = {:.5f}%'.format(mean_loss, mean_accuracy))
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')


def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', required=True,
                        help='Path to model weight file')
    parser.add_argument('-data_pkl', required=True,
                        help='Pickle file with both instances and vocabulary.')
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=5)
    parser.add_argument('-max_seq_len', type=int, default=100)
    parser.add_argument('-no_cuda', action='store_true')
    
    parser.add_argument('-batch_size', type=int, default=1)

    # TODO: Translate bpe encoded files 
    #parser.add_argument('-src', required=True,
    #                    help='Source sequence to decode (one line per sequence)')
    #parser.add_argument('-vocab', required=True,
    #                    help='Source sequence to decode (one line per sequence)')
    # TODO: Batch translation
    #parser.add_argument('-batch_size', type=int, default=30,
    #                    help='Batch size')
    #parser.add_argument('-n_best', type=int, default=1,
    #                    help="""If verbose is set, will output the n_best
    #                    decoded sentences""")

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    device = torch.device('cuda' if opt.cuda else 'cpu')
    test_data = prepare_dataloaders(opt, device)

    model=load_model(opt, device),
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001)

    eval(model, test_data, loss_function, optimizer, device, opt)

    print('[Info] Finished.')

if __name__ == "__main__":
    '''
    Usage: python translate.py -model trained.chkpt -data multi30k.pt -no_cuda
    '''
    main()
