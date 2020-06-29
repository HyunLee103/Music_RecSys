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


## target score(sparse matrix) 만들기
meta = pd.read_json('meta.json',orient='table')

train, test = train_test_split(meta, test_size=0.2, shuffle=True, random_state=1003)

train['istrain'] = 1
test['istrain'] = 0

n_train = len(train)
n_test = len(test)

# train + test
plylst = pd.concat([train,test], ignore_index=True)
plylst["nid"] = range(n_train+n_test)

# id <-> nid
plylst_id_nid = dict(zip(plylst["id"],plylst["nid"]))
plylst_nid_id = dict(zip(plylst["nid"],plylst["id"]))

plylst_tag = plylst['tags'][0:3]
for tgs in plylst_tag:
  print(tgs)
  for tg in tgs:
    print(tg)

tag_counter = Counter([tg for tgs in plylst_tag for tg in tgs])

tag_dict = {x:tag_counter[x] for x in tag_counter}

plylst_tag = plylst['tags']
tag_counter = Counter([tg for tgs in plylst_tag for tg in tgs]) # counter > 데이터 개수 셀 때 유용
tag_dict = {x:tag_counter[x] for x in tag_counter}

tag_id_tid = dict()
tag_tid_id = dict()

for i,t in enumerate(tag_dict):
  tag_id_tid[t] = i
  tag_tid_id[i] = t

n_tags = len(tag_dict)

plylst_song = plylst['songs']
song_counter = Counter([sg for sgs in plylst_song for sg in sgs])
song_dict = {x: song_counter[x] for x in song_counter}

song_id_sid = dict()
song_sid_id = dict()

for i, t in enumerate(song_dict):
  song_id_sid[t] = i
  song_sid_id[i] = t

n_songs = len(song_dict)

# plylst의 songs와 tags를 새로운 id로 변환하여 데이터프레임에 추가
plylst['songs_id'] = plylst['songs'].map(lambda x: [song_id_sid.get(s) for s in x if song_id_sid.get(s) != None]) # get ; key로 value 얻기
plylst['tags_id'] = plylst['tags'].map(lambda x: [tag_id_tid.get(t) for t in x if tag_id_tid.get(t) != None])

print(plylst['songs'].iloc[0:3])
print(plylst['songs_id'].iloc[0:3])

plylst_use = plylst[['id','istrain','nid','updt_date','songs_id','tags_id','plylst_title','song_gn_gnr_basket_flatten','artist_id_basket_flatten','artist_id_basket_count','song_gn_gnr_basket_count','tags']]
plylst_use.loc[:,'num_songs'] = plylst_use['songs_id'].map(len)
plylst_use.loc[:,'num_tags'] = plylst_use['tags_id'].map(len)
plylst_use = plylst_use.set_index('nid')

plylst_train = plylst_use.iloc[:n_train,:]
plylst_test = plylst_use.iloc[n_train:,:]

def rating(number):
  return [-math.log(x+1,2) +8.66 for x in range(number)]

# csr_matrix > (행,열)로 데이터 위치 표시
row = np.repeat(range(n_train),plylst_train['num_songs'])
col = [song for songs in plylst_train['songs_id'] for song in songs]
dat_series = plylst_train['num_songs'].map(rating)
dat = [y for x in dat_series for y in x]
train_songs_A = spr.csr_matrix((dat, (row, col)), shape=(n_train, n_songs))

row = np.repeat(range(n_train), plylst_train['num_tags'])
col = [tag for tags in plylst_train['tags_id'] for tag in tags]
dat_series = plylst_train['num_tags'].map(rating)
dat = [y for x in dat_series for y in x]
train_tags_A = spr.csr_matrix((dat, (row, col)), shape=(n_train, n_tags))

target = train_songs_A.T




## input X 불러와서 target과 합치기
X_train = pd.read_json('X_train.json',orient='table')


X_train = X_train.astype(float)
X_spr = spr.csr_matrix(X_train)
X_spr

full = hstack((X_spr,target.T))
full = full.tocsc()
full = full.T


## Modeling
def rmsle(y, y_pred):
  return np.sqrt(mean_squared_error(y, y_pred))

X = full[0:50].T.todense()
lst = list(range(92056))



## Training
for i in range(50,144724):
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
  joblib.dump(model_lgb,'./LGBM/{}_model'.format(i-50))
  print(i)
  # print(rmsle(Y, lgb_train_pred))   


