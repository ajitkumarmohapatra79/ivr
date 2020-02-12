from ..torch_core import *
from ..layers import *

__all__ = ['TabularModel', 'TabularExtModel']

class LinearRelu(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim) 

        # Non-linearity
        self.relu = nn.ReLU()

        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
        # Linear function  # LINEAR
        out = self.fc1(x)

        # Non-linearity  # NON-LINEAR
        out = self.relu(out)

        # Linear function (readout)  # LINEAR
        out = self.fc2(out)
        return out


class TabularExtModel(Module):
    "Basic model for tabular data."
    def __init__(self, vec_szs:ListSizes, emb_szs:ListSizes, n_cont:int, out_sz:int, layers:Collection[int], ps:Collection[float]=None,
                 emb_drop:float=0., y_range:OptRange=None, use_bn:bool=True, bn_final:bool=False):
        super().__init__()
        
        self.vecs = nn.ModuleList([LinearRelu(ni, nf) for ni,nf in vec_szs])
        self.n_vecs = sum(e.embedding_dim for e in self.vecs)
        self.vecs_drop = nn.Dropout(emb_drop)
        
        ps = ifnone(ps, [0]*len(layers))
        ps = listify(ps, layers)
        self.embeds = nn.ModuleList([embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(emb_drop)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb,self.n_cont,self.y_range = n_emb,n_cont,y_range
        sizes = self.get_sizes(layers, out_sz)
        actns = [nn.ReLU(inplace=True) for _ in range(len(sizes)-2)] + [None]
        layers = []
        for i,(n_in,n_out,dp,act) in enumerate(zip(sizes[:-1],sizes[1:],[0.]+ps,actns)):
            layers += bn_drop_lin(n_in, n_out, bn=use_bn and i!=0, p=dp, actn=act)
        if bn_final: layers.append(nn.BatchNorm1d(sizes[-1]))
        self.layers = nn.Sequential(*layers)

    def get_sizes(self, layers, out_sz):
        return [self.n_vecs + self.n_emb + self.n_cont] + layers + [out_sz]

    def forward(self, x_vec:Tensor, x_cat:Tensor, x_cont:Tensor) -> Tensor:
#         print('inside TabularExtModel forward')
#         b = 0
#         assert b != 0, "force an assertion to see who passed data_collate"        
        if self.n_vecs != 0:
            print('process n_vecs')
            # line up all vectors, use vec_szs to know the partitions
            x_vecs = [e(x_vec[:,i]) for i,e in enumerate(self.vecs)]
            x_vecs = torch.cat(x_vecs, 1)
            x_vecs = self.vecs_drop(x_vecs)            
        if self.n_emb != 0:
#             print('process n_embs')
#             print('x_cat shape={}'.format(x_cat.shape))
#             print('x_cat={}'.format(x_cat))
            x_emb = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
#             x = torch.cat(x, 1)
            x_emb = torch.cat([x_vecs, x_emb], 1) if self.n_vecs != 0 else torch.cat(x_emb, 1)
            x_emb = self.emb_drop(x_emb)
        if self.n_cont != 0:
#             print('process n_cont')
#             print('x_cont shape={}'.format(x_cont.shape))
#             print('x_cont={}'.format(x_cont))
            x_cont = self.bn_cont(x_cont)
            x = torch.cat([x_emb, x_cont], 1) if self.n_emb != 0 else torch.cat([x_vecs, x_cont], 1) if self.n_vecs != 0 else x_cont
        x = self.layers(x)
        if self.y_range is not None:
            x = (self.y_range[1]-self.y_range[0]) * torch.sigmoid(x) + self.y_range[0]
        return x


class TabularModel(Module):
    "Basic model for tabular data."
    def __init__(self, emb_szs:ListSizes, n_cont:int, out_sz:int, layers:Collection[int], ps:Collection[float]=None,
                 emb_drop:float=0., y_range:OptRange=None, use_bn:bool=True, bn_final:bool=False):
        super().__init__()
        ps = ifnone(ps, [0]*len(layers))
        ps = listify(ps, layers)
        self.embeds = nn.ModuleList([embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(emb_drop)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb,self.n_cont,self.y_range = n_emb,n_cont,y_range
        sizes = self.get_sizes(layers, out_sz)
        actns = [nn.ReLU(inplace=True) for _ in range(len(sizes)-2)] + [None]
        layers = []
        for i,(n_in,n_out,dp,act) in enumerate(zip(sizes[:-1],sizes[1:],[0.]+ps,actns)):
            layers += bn_drop_lin(n_in, n_out, bn=use_bn and i!=0, p=dp, actn=act)
        if bn_final: layers.append(nn.BatchNorm1d(sizes[-1]))
        self.layers = nn.Sequential(*layers)

    def get_sizes(self, layers, out_sz):
        return [self.n_emb + self.n_cont] + layers + [out_sz]

    def forward(self, x_cat:Tensor, x_cont:Tensor) -> Tensor:
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        x = self.layers(x)
        if self.y_range is not None:
            x = (self.y_range[1]-self.y_range[0]) * torch.sigmoid(x) + self.y_range[0]
        return x
