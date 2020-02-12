import pandas as pd
import pickle as pkl
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import re
from functools import reduce

import bokeh
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show, output_notebook, reset_output
from bokeh.palettes import d3
import bokeh.models as bmo
from bokeh.io import save, output_file
from bokeh.transform import factor_cmap, factor_mark

# +
import matplotlib as mpl
from MulticoreTSNE import MulticoreTSNE as TSNE
# init_notebook_mode(connected = True)
# color = sns.color_palette("Set2")
import warnings
warnings.filterwarnings("ignore")

import collections
# -

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
pd.options.display.max_rows = 999

# +
import sys

sys.path
# -

from fastai.basic_train import load_learner


class CatPlotter():
    
    def __init__(self):
#         with open('ivr_backup_data/cat_index_dict.pkl', 'rb') as cat_index_dict_file:
#             self.cat_index_dict = pkl.load(cat_index_dict_file)
            
        path = '/mnt/azmnt/code/Users/bho829/IVR/order'
        self.learn = load_learner(path)
        
        with open('ivr_backup_data/cat_dict.pkl', 'rb') as cat_dict_file:
            self.cat_dict = pkl.load(cat_dict_file)
            
        self.inner_train_df = pd.read_csv('ivr_backup_data/inner_train_df.csv')

    def plot_cat(self,cat_var):
        embed_index = self.cat_dict[cat_var]
        cat_weights = self.learn.model.embeds[embed_index].weight.data.cpu().numpy()
        print('weight len={}'.format(len(cat_weights)))
        
        self.inner_train_df[cat_var] = self.inner_train_df[cat_var].astype('category').cat.as_ordered()
        cat_keys = np.concatenate([['#na#'],self.inner_train_df[cat_var].cat.categories.values])
        print('cat_keys len={}'.format(len(cat_keys)))
        
        tsne_model = TSNE(n_jobs=4,
            perplexity=80,
            early_exaggeration=4, # Trying out exaggeration trick
            n_components=2,
            verbose=1,
            random_state=2018,
            n_iter=1000)

        tsne_tfidf = tsne_model.fit_transform(np.array(cat_weights))

        tsne_tfidf_df = pd.DataFrame(data=tsne_tfidf, columns=["x", "y"])

        tsne_tfidf_df["cat_var"] = cat_keys
        tsne_tfidf_df['cluster'] = cat_var    
        
        source = ColumnDataSource(data = dict(x = tsne_tfidf_df["x"], 
                  y = tsne_tfidf_df["y"],
                  # color = colormap[clean_data['topics_code'].values.tolist()],
                  cat_var = tsne_tfidf_df["cat_var"],
                  target = tsne_tfidf_df["cluster"]
                  ))        
        
        LABELS = [cat_var]
        
        output_notebook()
        plot_tfidf = bp.figure(plot_width = 800, plot_height = 700, 
                               title = "T-SNE applied to Model1 Entity Embedding",
                               tools = "pan, wheel_zoom, box_zoom, reset, hover, previewsave",
                               x_axis_type = None, y_axis_type = None, min_border = 1)

        plot_tfidf.scatter(x = "x", 
                           y = "y", 
                           legend = "target",
                           source = source,
                           alpha = 0.7, 
                           marker=factor_mark('target', ['circle'], LABELS),
                           color=factor_cmap('target', palette=['blue'], factors=LABELS))


        hover = plot_tfidf.select(dict(type = HoverTool))
        hover.tooltips = {
                          "cat_var": "@cat_var",
                            "target": "@target"
                          }

        show(plot_tfidf)       


