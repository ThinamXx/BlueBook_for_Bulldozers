#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Downloading all Libraries

# In[2]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import math
import graphviz
import scipy
# import ggplot

# from pandas_summary import DataFrameSummary
from pandas.api.types import is_string_dtype, is_numeric_dtype
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn.tree import export_graphviz
from sklearn.ensemble import forest
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.cluster import hierarchy as hc
from pdpbox import pdp
from plotnine import *
# from concurrent.futures import ProcessPoolExecutor

from sklearn import metrics


# ### Loading the Dataset

# In[3]:


def proc_df(df, y_fld=None, skip_flds=None, ignore_flds=None, do_scale=False, na_dict=None, prepoc_fn=None, max_n_cat=None,
           subset=None, mapper=None):
    if not ignore_flds: ignore_flds=[]
    if not skip_flds: skip_flds=[]
    if subset:
        df = get_sample(df, subset)
    else:
        df = df.copy()
    ignored_flds = df.loc[:, ignore_flds]
    df.drop(ignore_flds, axis=1, inplace=True)
    if prepoc_fn: prepoc_fn(df)
    if y_fld is None: y=None
    else:
        if not is_numeric_dtype(df[y_fld]): df[y_fld] = pd.Categorical(df[y_fld]).codes
        y = df[y_fld].values
        skip_flds += [y_fld]
    df.drop(skip_flds, axis=1, inplace=True)
    
    if na_dict is None: na_dict = {}
    else: na_dict = na_dict.copy()
    na_dict_initial = na_dict.copy()
    for n, c in df.items(): na_dict = fix_missing(df, c, n, na_dict)
    if len(na_dict_initial.keys()) > 0:
        df.drop([a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)
    if do_scale: mapper = scale_vars(df, mapper)
    for n, c in df.items(): numericalize(df, c, n, max_n_cat)
    df = pd.get_dummies(df, dummy_na=True)
    df = pd.concat([ignored_flds, df], axis=1)
    res = [df, y, na_dict]
    if do_scale: res = res + [mapper]
    return res


# In[4]:


def fix_missing(df, col, name, na_dict):
    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (name in na_dict): 
            df[name + '_na'] = pd.isnull(col)
            filler = na_dict[name] if name in na_dict else col.median()
            df[name] = col.fillna(filler)
            na_dict[name] = filler
    return na_dict


# In[5]:


def numericalize(df, col, name, max_n_cat):
    if not is_numeric_dtype(col) and (max_n_cat is None or col.nunique()>max_n_cat):
        df[name] = col.cat.codes+1


# In[6]:


def get_sample(df, n):
    idxs = sorted(np.random.permutation(len(df))[:n])
    return df.iloc[idxs].copy()


# In[7]:


def set_rf_samples(n):
    forest._generate_sample_indices = (lambda rs, n_samples:
                                      forest.check_random_state(rs).randit(0, n_samples, n))


# In[8]:


def reset_rf_samples():
    forest._generate_sample_indices = (lambda rs, n_samples:
                                      forest.check_random_state(rs).randit(0, n_samples, n_samples))


# In[9]:


def split_vals(a, n):
    return a[:n].copy(), a[n:].copy()


# In[10]:


def rmse(x, y):
    return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train),
          rmse(m.predict(X_valid), y_valid),
          m.score(X_train, y_train),
          m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'):
        res.append(m.oob_score_)
    print(res)


# In[11]:


PATH = "C:\\Users\\DELL\\Blue Book for Bulldozers\\Train.csv"
df_raw = pd.read_feather('tmp/raw')


# In[12]:


df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice')


# In[13]:


n_valid = 12000
n_trn = len(df_trn) - n_valid

X_train, X_valid = split_vals(df_trn, n_trn)
y_train, y_valid = split_vals(y_trn, n_trn)
raw_train, raw_valid = split_vals(df_raw, n_trn)


# In[14]:


x_sub = X_train[['YearMade', 'MachineHoursCurrentMeter']]


# ### Random Forest @ Tree Ensemble

# In[15]:


class TreeEnsemble():
    
    def __init__(self, x, y, n_trees, sample_sz, min_leaf=5):
        np.random.seed(42)
        self.x = x
        self.y = y
        self.sample_sz = sample_sz
        self.min_leaf = min_leaf
        self.trees = [self.create_tree() for i in range(n_trees)]
    
    def create_tree(self):
        rnd_idxs = np.random.permutation(len(self.y))[:self.sample_sz]
        return DecisionTree(self.x.iloc[rnd_idxs], self.y[rnd_idxs], min_leaf=self.min_leaf)
    
    def predict(self, x):
        return np.mean([t.predict(x) for t in self.trees], axis=0)


# In[16]:


class DecisionTree():
    
    def __init__(self, x, y, idxs=None, min_leaf=5):
        if idxs is None:
            idxs = np.arange(len(y))
        self.x, self.y = x, y
        self.idxs =idxs
        self.min_leaf = min_leaf
        self.n, self.c = len(idxs), x.shape[1]
        self.val = np.mean(y[idxs])
        self.score = float('inf')
        self.find_varsplit()
        
    def find_varsplit(self):
        for i in range(self.c):
            self.find_better_split(i)
            
    def fing_better_split(self, var_idx):
        pass
    
    @property
    def split_name(self):
        return self.x.columns[self.var_idx]
    
    @property
    def split_col(self):
        return self.x.values[self.idxs, self.var_idx]
    
    @property
    def is_leaf(self):
        return self.score == float('inf')
    
    def __repr__(self):
        s = f'n: {self.n}; val: {self.val}'
        if not self.is_leaf:
            s += f'score: {self.score}; split: {self.split}; var: {self.split_name}'
            return s
        
        

