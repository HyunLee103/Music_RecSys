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

parser = argparse.ArgumentParser(description="score regression for RE_RANK",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--mode", default="song", choices=["song", "tag"], type=str, dest="mode")
args = parser.parse_args()

if args.mode == 'tag':
  full = mmread('rerank_tag_full.mtx')
  full = full.tocsc()

  ## Modeling
  def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


  file=open("tid2tag","rb") 
  tid2tag=pickle.load(file)


  full = full.T
  X = full[0:64].T.todense()
  lst = list(range(115071))


  ## Training
  for i in range(64,512):
    tag = full[i]
    y = tag.data

    neg_1 = list(set(lst) - set(tag.indices))
    k = round(len(y)*1)
    neg_idx = random.choices(neg_1, k=k)

    X_input = X[neg_idx+list(tag.indices), :]

    Y = np.concatenate((y, np.zeros(k)))
    model_lgb = lgb.LGBMRanker(g)

    model_lgb.fit(X_input, Y)

    # lgb_train_pred = model_lgb.predict(X_input)
    joblib.dump(model_lgb,'./tag_regression_4/{}_model'.format(tid2tag.get((i-64))))
    print(i-64)
    # print(rmsle(Y, lgb_train_pred))   

elif args.mode == 'song':
  full = mmread('rerank_song_last.mtx')
  full = full.tocsc()

  file=open("sid2id","rb") 
  sid2id=pickle.load(file)

  full = full.T
  X = full[0:106].T.todense()
  lst = list(range(117094))
  
  ## Training
  for i in tqdm(range(106,20189)):
    song = full[i]
    y = song.data

    # neg_1 = list(set(lst) - set(song.indices))
    # k = round(len(y)*1)
    # neg_idx = random.choices(neg_1, k=k)

    X_input = X[list(song.indices), :]

    # Y = np.concatenate((y, np.zeros(k)))
    model_lgb = lgb.LGBMRegressor(n_estimators=600)

    model_lgb.fit(X_input, y)

    # lgb_train_pred = model_lgb.predict(X_input)
    joblib.dump(model_lgb,'./song_regression_last/{}_model'.format(sid2id.get((i-106))))
    print(i-106)
    # print(rmsle(Y, lgb_train_pred))   


