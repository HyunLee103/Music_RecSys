import numpy as np
import pandas as pd

import scipy.sparse as spr
import pickle
from scipy.sparse import hstack
from collections import Counter

from scipy.io import mmwrite
from scipy.io import mmread

# -*- coding: utf-8 -*-
import io
import os
import json
import distutils.dir_util
from collections import Counter
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
import numpy as np
import pandas as pd
import json
import scipy.sparse as spr
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import math
import joblib
import random

import io
import os
import distutils.dir_util
from collections import Counter
from scipy.sparse import coo_matrix, hstack
from scipy.io import mmread

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.datasets import make_regression
from sklearn.model_selection import GridSearchCV
from scipy.io import mmwrite
from scipy.io import mmread
import pickle 

import argparse


def write_json(data, fname):
    def _conv(o):
        if isinstance(o, (np.int64, np.int32)):
            return int(o)
        raise TypeError

    parent = os.path.dirname(fname)
    distutils.dir_util.mkpath("./arena_data/" + parent)
    with io.open("./arena_data/" + fname, "w", encoding="utf-8") as f:
        json_str = json.dumps(data, ensure_ascii=False, default=_conv)
        f.write(json_str)

def load_json(fname):
    with open(fname, encoding="utf-8") as f:
        json_obj = json.load(f)

    return json_obj


class PlyEmbedding:
    """
    def __init__
     - data: for song based embedding
    
    def make_s2v
     - params: same with Word2Vec params
     
    def make_d2v
     - vector_size: vector size of the Doc2Vec model
     
    def song_based
     - mode: 's2v'(Word2Vec base) or 'd2v'(Doc2Vec base)
     - by:   'mean' or 'sum'
     - keyedvector: True -> return Word2Vec type
                    False-> return (id+titles, vectors) : list
                    플레이리스트 벡터로 근처 플레이리스트 찾을 땐 True
                    클러스터링 학습을 하고 싶을 땐 False
    """
    
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.s2v = None
        self.d2v = None
        print("Data length:", len(data))
        
    def make_s2v(self, min_count=5, size=100, window=5, sg=0):
        songs = [list(map(str, x['songs'])) for x in self.data if len(x['songs']) > 1]
        print("Original data length: ", len(self.data))
        print("Playlist count after remove 0 or 1: ", len(songs))
        print("Traning song2vec...")
        self.s2v = Word2Vec(songs, size=size, window=window,
                            min_count=min_count, sg=sg)
        print("done.")
        
    def get_song2vec(self):
        return self.s2v
        
    def make_d2v(self, vector_size=100):
        doc = [TaggedDocument(list(map(str, x['songs'])),
                        ['(' + str(x['id']) + ') ' + x['plylst_title']]) for x in self.data]
        
        print("Training Doc2Vec...")
        self.d2v = Doc2Vec(doc, vector_size=vector_size, workers=4)
        print('done')
        
    def get_doc2vec(self):
        return self.d2v
    
    def song_based(self, mode='s2v', by='mean', keyedvector=True):
        if mode == 's2v':
            if not self.s2v:
                print("Song2Vec not exist.\nRun make_s2v first.")
                return
        elif mode == 'd2v':
            if not self.d2v:
                print("Doc2Vec not exist.\nRun make_d2v first.")
                return
        else:
            print("mode gets 's2v' or 'd2v'")
            
        if not by in ['mean', 'sum']:
            raise RuntimeError("'by' gets 'mean' or 'sum'")
        
        ply_id = []
        ply_vec = []
        
        for p in tqdm(self.data):
            if by == 'mean':
                tmp = []
            else:
                tmp = 0
            for song in p['songs']:
                try:
                    if by == 'mean':
                        if mode == 's2v':
                            tmp.append(self.s2v.wv.get_vector(str(song)))
                        else:
                            tmp.append(self.d2v.wv.get_vector(str(song)))
                    else:
                        if mode == 's2v':
                            tmp += self.s2v.wv.get_vector(str(song))
                        else:
                            tmp += self.d2v.wv.get_vector(str(song))
                except KeyError:
                    pass
            if by == 'mean':
                if tmp != []:
                    ply_id.append('(' + str(p['id']) + ') ' + p['plylst_title'])
                    ply_vec.append(np.mean(tmp, axis=0))
            else:
                if type(tmp) != int:
                    ply_id.append('(' + str(p['id']) + ') ' + p['plylst_title'])
                    ply_vec.append(tmp)
        
        print("Original data length: ", len(self.data))
        print("Embedded data length: ", len(ply_id))
        
        if not keyedvector:
            return ply_id, ply_vec
        
        out = WordEmbeddingsKeyedVectors(vector_size=100)
        out.add(ply_id, ply_vec)
        
        return out
        
    def by_autoencoder(self):
        pass




def debug_json(r):
    print(json.dumps(r, ensure_ascii=False, indent=4))


def remove_seen(seen, l):
    seen = set(seen)
    return [x for x in l if not (x in seen)]


def most_popular(playlists, col, topk_count):
    c = Counter()

    for doc in playlists:
        c.update(doc[col])

    topk = c.most_common(topk_count)
    return c, [k for k, v in topk]