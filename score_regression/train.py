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
  full = mmread('tag_full.mtx')
  full = full.tocsc()

  ## Modeling
  def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


  file=open("tid2tag","rb") 
  tid2tag=pickle.load(file)


  full = full.T
  X = full[0:73].T.todense()
  lst = list(range(115071))


  ## Training
  for i in range(73,29703):
    tag = full[i]
    y = tag.data

    neg_1 = list(set(lst) - set(tag.indices))
    k = round(len(y)*1.5)
    neg_idx = random.choices(neg_1, k=k)

    X_input = X[neg_idx+list(tag.indices), :]

    Y = np.concatenate((y, np.zeros(k)))
    model_lgb = lgb.LGBMRegressor(n_estimators=500)

    model_lgb.fit(X_input, Y)

    # lgb_train_pred = model_lgb.predict(X_input)
    joblib.dump(model_lgb,'./tag_regression_2/{}_model'.format(tid2tag.get((i-73))))
    print(i-73)
    # print(rmsle(Y, lgb_train_pred))   

elif args.mode == 'song':
  full = mmread('song_full.mtx')
  full = full.tocsc()

  ## Modeling
  def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


  file=open("sid2id","rb") 
  sid2id=pickle.load(file)


  full = full.T
  X = full[0:73].T.todense()
  lst = list(range(115071))


  ## Training
  for i in range(73,145000):
    song = full[i]
    y = song.data

    neg_1 = list(set(lst) - set(song.indices))
    k = round(len(y)*1.5)
    neg_idx = random.choices(neg_1, k=k)

    X_input = X[neg_idx+list(song.indices), :]

    Y = np.concatenate((y, np.zeros(k)))
    model_lgb = lgb.LGBMRegressor(n_estimators=300)

    model_lgb.fit(X_input, Y)

    # lgb_train_pred = model_lgb.predict(X_input)
    joblib.dump(model_lgb,'./song_regression_2/{}_model'.format(sid2id.get((i-73))))
    print(i-73)
    # print(rmsle(Y, lgb_train_pred))   



