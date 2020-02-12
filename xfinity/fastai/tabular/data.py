"Data loading pipeline for structured data support. Loads from pandas DataFrame"
from ..torch_core import *
from .transform import *
from ..basic_data import *
from ..data_block import *
from pandas.api.types import is_numeric_dtype, is_categorical_dtype

__all__ = ['TabularDataBunch', 'TabularLine', 'TabularList', 'TabularProcessor']

OptTabTfms = Optional[Collection[TabularProc]]

#def emb_sz_rule(n_cat:int)->int: return min(50, (n_cat//2)+1)
def emb_sz_rule(n_cat:int)->int: return min(600, round(1.6 * n_cat**0.56))

def def_emb_sz(classes, n, sz_dict=None):
    "Pick an embedding size for `n` depending on `classes` if not given in `sz_dict`."
    sz_dict = ifnone(sz_dict, {})
    n_cat = len(classes[n])
    sz = sz_dict.get(n, int(emb_sz_rule(n_cat)))  # rule of thumb
    return n_cat,sz

def vec_sz_rule(n_cat:int)->int: return min(600, round(1.6 * n_cat**0.56))


def def_vec_sz(classes, n, sz_dict=None):
    "Pick an embedding size for `n` depending on `classes` if not given in `sz_dict`."
    sz_dict = ifnone(sz_dict, {})
    n_vec_dim = len(classes[n])
    sz = sz_dict.get(n, int(vec_sz_rule(n_vec_dim)))  # rule of thumb
    return n_vec_dim,sz


class TabularLine(ItemBase):
    "Basic item for tabular data."
    def __init__(self, vecs, cats, conts, classes, names):
        self.vecs,self.cats,self.conts,self.classes,self.names = vecs,cats,conts,classes,names
        self.data = [tensor(vecs), tensor(cats), tensor(conts)]

    def __str__(self):
        res = ''
        for c, n in zip(self.vecs, self.names[:len(self.vecs)]):
            res += f"{n} {c:.4f}; "
        for c, n in zip(self.cats, self.names[len(self.vecs):len(self.cats)]):
            res += f"{n} {(self.classes[n][c])}; "
        for c,n in zip(self.conts, self.names[len(self.cats):]):
            res += f'{n} {c:.4f}; '
        return res

class TabularProcessor(PreProcessor):
    "Regroup the `procs` in one `PreProcessor`."
    def __init__(self, ds:ItemBase=None, procs=None):
        procs = ifnone(procs, ds.procs if ds is not None else None)
        self.procs = listify(procs)

    def process_one(self, item):
        df = pd.DataFrame([item,item])
        for proc in self.procs: proc(df, test=True)
        if len(self.vec_names) != 0:
            vec_codes = np.stack([c.astype('float32').values for n,c in df[self.vec_names].items()], 1).astype(np.int64) + 1
        else: vec_codes = [[]]
        if len(self.cat_names) != 0:
            codes = np.stack([c.cat.codes.values for n,c in df[self.cat_names].items()], 1).astype(np.int64) + 1
        else: codes = [[]]
        if len(self.cont_names) != 0:
            conts = np.stack([c.astype('float32').values for n,c in df[self.cont_names].items()], 1)
        else: conts = [[]]
        classes = None
        col_names = list(df[self.vec_names].columns.values) + list(df[self.cat_names].columns.values) + list(df[self.cont_names].columns.values)
        return TabularLine(vec_codes[0], codes[0], conts[0], classes, col_names)

    def process(self, ds):
        if ds.inner_df is None:
            ds.vec_names, ds.classes,ds.cat_names,ds.cont_names = self.vec_names, self.classes,self.cat_names,self.cont_names
            ds.col_names = self.vec_names + self.cat_names + self.cont_names
            ds.preprocessed = True
            return
        for i,proc in enumerate(self.procs):
            if isinstance(proc, TabularProc): proc(ds.inner_df, test=True)
            else:
                #cat and cont names may have been changed by transform (like Fill_NA)
                proc = proc(ds.vec_names, ds.cat_names, ds.cont_names)
                proc(ds.inner_df)
                ds.vec_names,ds.cat_names,ds.cont_names = proc.vec_names,proc.cat_names,proc.cont_names
                self.procs[i] = proc
        self.vec_names,self.cat_names,self.cont_names = ds.vec_names,ds.cat_names,ds.cont_names
        if len(ds.vec_names) != 0:
            ds.vec_codes = np.stack([c.astype('float32').values for n,c in ds.inner_df[ds.vec_names].items()], 1)
            vec_cols = list(ds.inner_df[ds.vec_names].columns.values)
        else: ds.vec_codes,vec_cols = None,[]
        if len(ds.cat_names) != 0:
            ds.codes = np.stack([c.cat.codes.values for n,c in ds.inner_df[ds.cat_names].items()], 1).astype(np.int64) + 1
            self.classes = ds.classes = OrderedDict({n:np.concatenate([['#na#'],c.cat.categories.values])
                                      for n,c in ds.inner_df[ds.cat_names].items()})
            cat_cols = list(ds.inner_df[ds.cat_names].columns.values)
        else: 
            ds.codes,ds.classes,self.classes,cat_cols = None,None,None,[]
        if len(ds.cont_names) != 0:
            ds.conts = np.stack([c.astype('float32').values for n,c in ds.inner_df[ds.cont_names].items()], 1)
            cont_cols = list(ds.inner_df[ds.cont_names].columns.values)
        else: ds.conts,cont_cols = None,[]
        ds.col_names = vec_cols + cat_cols + cont_cols
        ds.preprocessed = True

class TabularDataBunch(DataBunch):
    "Create a `DataBunch` suitable for tabular data."
    @classmethod
    def from_df(cls, path, df:DataFrame, dep_var:str, valid_idx:Collection[int], procs:OptTabTfms=None,
                vec_names:OptStrList=None, cat_names:OptStrList=None, cont_names:OptStrList=None, classes:Collection=None, 
                test_df=None, bs:int=64, val_bs:int=None, num_workers:int=defaults.cpus, dl_tfms:Optional[Collection[Callable]]=None, 
                device:torch.device=None, collate_fn:Callable=data_collate, no_check:bool=False)->DataBunch:
        "Create a `DataBunch` from `df` and `valid_idx` with `dep_var`. `kwargs` are passed to `DataBunch.create`."
        vec_names = ifnone(vec_names, []).copy()
        cat_names = ifnone(cat_names, []).copy()
        cont_names = ifnone(cont_names, list(set(df)-set(cat_names)-{dep_var}))
        procs = listify(procs)
        src = (TabularList.from_df(df, path=path, vec_names=vec_names, cat_names=cat_names, cont_names=cont_names, procs=procs)
                           .split_by_idx(valid_idx))
#         print("src type={}".type(src))
        src = src.label_from_df(cols=dep_var) if classes is None else src.label_from_df(cols=dep_var, classes=classes)
#         print("after label_from src type={}".type(src))
        if test_df is not None: src.add_test(TabularList.from_df(test_df, vec_names=vec_names, cat_names=cat_names, cont_names=cont_names,
                                                                 processor = src.train.x.processor))
        return src.databunch(path=path, bs=bs, val_bs=val_bs, num_workers=num_workers, device=device, 
                             collate_fn=collate_fn, no_check=no_check)

class TabularList(ItemList):
    "Basic `ItemList` for tabular data."
    _item_cls=TabularLine
    _processor=TabularProcessor
    _bunch=TabularDataBunch
    def __init__(self, items:Iterator, vec_names:OptStrList=None, cat_names:OptStrList=None, cont_names:OptStrList=None,
                 procs=None, **kwargs)->'TabularList':
        super().__init__(range_of(items), **kwargs)
        #dataframe is in inner_df, items is just a range of index
        if vec_names is None:  vec_names = []
        if cat_names is None:  cat_names = []
        if cont_names is None: cont_names = []
        self.vec_names,self.cat_names,self.cont_names,self.procs = vec_names, cat_names,cont_names,procs
        self.copy_new += ['vec_names', 'cat_names', 'cont_names', 'procs']
        self.preprocessed = False

    @classmethod
    def from_df(cls, df:DataFrame, vec_names:OptStrList=None, cat_names:OptStrList=None, cont_names:OptStrList=None, procs=None, **kwargs)->'ItemList':
        "Get the list of inputs in the `col` of `path/csv_name`."
        return cls(items=range(len(df)), vec_names=vec_names, cat_names=cat_names, cont_names=cont_names, procs=procs, inner_df=df.copy(), **kwargs)

    def get(self, o):
#         print('TabularList get({})'.format(o))
#         print('self.preprocessed={}'.format(self.preprocessed))
        if not self.preprocessed: return self.inner_df.iloc[o] if hasattr(self, 'inner_df') else self.items[o]
#         print('TabularList get {} '.format(o))
#         print('self items={}'.format(self.items))
#         print('self vec_codes={}'.format(self.vec_codes))
        vec_codes = [] if self.vec_codes is None else self.vec_codes[o]
        codes = [] if self.codes is None else self.codes[o]
        conts = [] if self.conts is None else self.conts[o]
        return self._item_cls(vec_codes, codes, conts, self.classes, self.col_names)

    def get_emb_szs(self, sz_dict=None):
        "Return the default embedding sizes suitable for this data or takes the ones in `sz_dict`."
        return [def_emb_sz(self.classes, n, sz_dict) for n in self.cat_names]

    def get_vec_szs(self, sz_dict=None):
        "Return the default list embedding sizes suitable for this data or takes the ones in `sz_dict`."
        return [def_vec_sz(self.classes, n, sz_dict) for n in self.vec_names]

    def reconstruct(self, t:Tensor):
        return self._item_cls(t[0], t[1], self.classes, self.col_names)

    def show_xys(self, xs, ys)->None:
        "Show the `xs` (inputs) and `ys` (targets)."
        from IPython.display import display, HTML
        items,names = [], xs[0].names + ['target']
        for i, (x,y) in enumerate(zip(xs,ys)):
            res = []
            vecs = x.vecs if len(x.vecs.size()) > 0 else []
            cats = x.cats if len(x.cats.size()) > 0 else []
            conts = x.conts if len(x.conts.size()) > 0 else []
            for c, n in zip(cats, x.names[:len(cats)]):
                res.append(x.classes[n][c])
            res += [f'{c:.4f}' for c in conts] + [y]
            items.append(res)
        items = np.array(items)
        df = pd.DataFrame({n:items[:,i] for i,n in enumerate(names)}, columns=names)
        with pd.option_context('display.max_colwidth', -1):
            display(HTML(df.to_html(index=False)))

    def show_xyzs(self, xs, ys, zs):
        "Show `xs` (inputs), `ys` (targets) and `zs` (predictions)."
        from IPython.display import display, HTML
        items,names = [], xs[0].names + ['target', 'prediction']
        for i, (x,y,z) in enumerate(zip(xs,ys,zs)):
            res = []
            vecs = x.vecs if len(x.vecs.size()) > 0 else []
            cats = x.cats if len(x.cats.size()) > 0 else []
            conts = x.conts if len(x.conts.size()) > 0 else []
            for c, n in zip(cats, x.names[:len(cats)]):
                res.append(str(x.classes[n][c]))
            res += [f'{c:.4f}' for c in conts] + [y, z]
            items.append(res)
        items = np.array(items)
        df = pd.DataFrame({n:items[:,i] for i,n in enumerate(names)}, columns=names)
        with pd.option_context('display.max_colwidth', -1):
            display(HTML(df.to_html(index=False)))


