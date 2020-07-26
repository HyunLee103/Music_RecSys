"""
song_regression_last(over_50) : 2만곡 대상

--input feature--

cnt : 2
song_embed : 32 (over_50 대상)
tag_embed : 32
genre : 30
season : 4
year : 6
"""
from utils import *
import scipy.sparse as spr
from scipy.sparse import hstack
from collections import Counter
from tqdm import tqdm

from scipy.io import mmwrite
from scipy.io import mmread

import gensim
import random
import io
import os
import json
import distutils.dir_util
from collections import Counter
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
import joblib
import random

from scipy.sparse import coo_matrix, hstack

import pickle 


## util function
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


## load data
song_meta = pd.read_json('data/meta/song_meta.json', typ = 'frame',encoding='utf-8')
train = pd.read_json('data/meta/train.json', typ = 'frame',encoding='utf-8')
val = pd.read_json('data/meta/val.json', typ = 'frame',encoding='utf-8')
genre_gn_all = pd.read_json('data/meta/genre_gn_all.json', typ = 'series')
val_t2r = load_json('val_t2r.json')
val_t2r = pd.DataFrame(val_t2r)
test = load_json('data/meta/test.json')
test = pd.DataFrame(test)


## over_50로 song 처리
full = pd.concat([train[['tags','id','songs','updt_date']],val[['tags','id','songs','updt_date']],test[['tags','id','songs','updt_date']]])
# full.to_json('full.json',orient='records') # song embed에 사용

plylst_song_map = full[['id', 'songs']]

plylst_song_map_unnest = np.dstack(
    (
        np.repeat(plylst_song_map.id.values, list(map(len, plylst_song_map.songs))), 
        np.concatenate(plylst_song_map.songs.values)
    )
)

plylst_song_map = pd.DataFrame(data = plylst_song_map_unnest[0], columns = plylst_song_map.columns)
plylst_song_map['id'] = plylst_song_map['id'].astype(int)
plylst_song_map['songs'] = plylst_song_map['songs'].astype(int)

del plylst_song_map_unnest

value = plylst_song_map['songs'].value_counts() > 50

df = plylst_song_map['songs'].value_counts().rename_axis('unique_values').to_frame('counts')
df = df.reset_index() ; df.head(3)
over_5 = df[df['counts']>50]

requred_song = over_5['unique_values'].tolist()
requred_song = set(list(map(int, requred_song)))
len(requred_song)

df = pd.DataFrame(columns=['over50_songs'])
full = pd.concat([full,df])

for i in range(full.shape[0]):     
    result = set(full['songs'].iloc[i]) & requred_song
    full['over50_songs'].iloc[i] = list(result)
del full['songs']
full.rename(columns = {'over50_songs' : 'songs'}, inplace = True)
len(requred_song)

full = full[full['songs'].apply(lambda x : x != []) & full['tags'].apply(lambda x : x != [])]



## score matrix 만들기(sparse)
train = full
n_train = len(train)

plylst = train
plylst["nid"] = range(n_train)

plylst_id_nid = dict(zip(plylst["id"],plylst["nid"]))   
plylst_nid_id = dict(zip(plylst["nid"],plylst["id"]))   
             

plylst_song = plylst['songs']          
song_counter = Counter([sg for sgs in plylst_song for sg in sgs])
song_dict = {x: song_counter[x] for x in song_counter}

song_id_sid = dict()
song_sid_id = dict()
for i, t in enumerate(song_dict):       
    song_id_sid[t] = i
    song_sid_id[i] = t

n_songs = len(song_dict)
plylst['songs_id'] = plylst['songs'].map(lambda x: [song_id_sid.get(s) for s in x if song_id_sid.get(s) != None]) 

sid2id = {v: k for k, v in song_id_sid.items()}
file=open("data/sid2id","wb") 
pickle.dump(sid2id,file) 
file.close()

plylst_use = plylst
plylst_use.loc[:,'num_songs'] = plylst_use['songs_id'].map(len)
plylst_use = plylst_use.set_index('nid')

plylst_train = plylst_use

def rating(number):
  return [-math.log(x+1,2) +8.66 for x in range(number)]

row = np.repeat(range(n_train),plylst_train['num_songs'])
col = [song for songs in plylst_train['songs_id'] for song in songs]
dat_series = plylst_train['num_songs'].map(rating)
dat = [y for x in dat_series for y in x]
train_songs_A = spr.csr_matrix((dat, (row, col)), shape=(n_train, n_songs))



## option : t2r에서 비어있는 song, tag most popular로 채우기
train = load_json('data/meta/train.json')
_, pop_songs = most_popular(train, 'songs', 100)
_, pop_tags = most_popular(train, 'tags', 10)

no = full[full['id'].isin(set(full['id']) - set(meta['id']))]

def put_most_popular(pop):
    return pop_tags

no['songs'] = no['songs'].apply(put_most_popular)
no['tags'] = no['tags'].apply(put_most_popular)


def random_crop(x):
    k = random.randrange(0,50)
    return x[k:k+30]

def random_crop_t(x):
    k = random.randrange(0,7)
    return x[k:k+3]


no['songs'] = no['songs'].apply(random_crop)
no['tags'] = no['tags'].apply(random_crop_t)

yes = full[full['id'].isin(set(meta['id']))]

full = pd.concat([yes,no])



## meta 정보 concat
plylst_song_map = full[['id', 'songs']]

plylst_song_map_unnest = np.dstack(
    (
        np.repeat(plylst_song_map.id.values, list(map(len, plylst_song_map.songs))), 
        np.concatenate(plylst_song_map.songs.values)
    )
)

plylst_song_map = pd.DataFrame(data = plylst_song_map_unnest[0], columns = plylst_song_map.columns)
plylst_song_map['id'] = plylst_song_map['id'].astype(str)
plylst_song_map['songs'] = plylst_song_map['songs'].astype(str)

del plylst_song_map_unnest

plylst_song_map = plylst_song_map.astype(float)
# plylst_song_map = plylst_song_map.astype(int)

plylst_song_map = pd.merge(plylst_song_map,song_meta,how='left',left_on='songs',right_index=True)
plylst_song_map = plylst_song_map.drop('id_y',axis=1)
plylst_song_map.rename(columns={'id_x':'id'},inplace=True)

plylst_meta = pd.DataFrame(full[['id','tags','updt_date']])

for column in plylst_song_map.columns[1:]:
  plylst_sum = pd.DataFrame(plylst_song_map.groupby('id')[column].apply(list))
  plylst_sum = plylst_sum.reset_index()

  plylst_sum['id'] = plylst_sum['id'].astype(str).astype(float)
  plylst_meta = pd.merge(plylst_meta,plylst_sum,left_on='id',right_on='id',how='inner')

list_columns = ['song_gn_dtl_gnr_basket','artist_id_basket','song_gn_gnr_basket','artist_name_basket']

for column in list_columns:
  plylst_meta[f'{column}_flatten'] = plylst_meta[column].map(lambda x : sum(x,[])) # 이중리스트 단일 리스트로. (list_columns의 column들이 이중리스트인 것들)
  plylst_meta[f'{column}_unique'] = plylst_meta[f'{column}_flatten'].map(lambda x : list(set(x))) # 리스트 > 집합 > 리스트로 unique한 값 남김
  plylst_meta[f'{column}_count'] = plylst_meta[f'{column}_unique'].map(lambda x : len(x)) # unique한 것 개수 세기

meta = plylst_meta[['id','tags','songs','song_gn_gnr_basket_flatten','artist_id_basket_count','song_gn_gnr_basket_count','updt_date']]


## genre 임배딩
X = meta
genre_gn_all = pd.read_json('data/meta/genre_gn_all.json', typ = 'series')
genre_gn_all = pd.DataFrame(genre_gn_all, columns = ['gnr_name']).reset_index().rename(columns = {'index' : 'gnr_code'})
gnr_code = genre_gn_all[genre_gn_all['gnr_code'].str[-2:] == '00']
code2idx = {code:i for i, code in gnr_code['gnr_code'].reset_index(drop=True).items()}
code2idx['GN9000'] = 30

def genre_cnt(x):
    counter = Counter(x)
    out = np.zeros(31)
    for gnr, cnt in counter.items():
        out[code2idx[gnr]] = cnt
    return out/len(x)

X_gn = pd.concat([X, pd.DataFrame(list(X['song_gn_gnr_basket_flatten'].apply(genre_cnt)))],axis=1)

X_gn = X_gn.add_prefix('gn_') 

X_gn.rename(columns = {'gn_id':'id','gn_plylst_title':'plylst_title','gn_song_gn_gnr_basket_flatten':'song_gn_gnr_basket_flatten','gn_artist_id_basket_flatten':'artist_id_basket_flatten',
                      'gn_artist_id_basket_count':'artist_id_basket_count','gn_song_gn_gnr_basket_count':'song_gn_gnr_basket_count','gn_tags':'tags','gn_year':'year','gn_month':'month',
                      'gn_season':'season','gn_year_section':'year_section'},inplace=True)

X_gn.head(3)

del X_gn['gn_30']
X_gn.rename(columns={'gn_songs':'songs'},inplace=True)




## song 임배딩
full_1 = load_json('full.json') # 앞에서 저장해둔 원래 full데이터로 song embedding(over_50 적용 X)
embed = PlyEmbedding(full_1)

embed.make_s2v(size=32)
m = embed.get_song2vec()
song_vector = m.wv

song = song_vector.vocab.keys()
song_vector_lst = [song_vector[v] for v in song]

from gensim.models import KeyedVectors
song_vector.save_word2vec_format('song2v_over50') 

# inference
song_vector = gensim.models.KeyedVectors.load_word2vec_format('song2v_over50')

def song_embed(x):
    tem = []
    for s in x:
        try:
            tem.append(song_vector.get_vector(s))
        except KeyError as e:
            pass
    if tem == []:
        return np.zeros(32)
    else:
        return np.mean(tem,axis=0)

songs = list(map(str, X_gn['songs']))



X_songembed = pd.concat([X_gn, pd.DataFrame(list(pd.Series(songs).apply(song_embed)))],axis =1)
X_songembed.rename(columns = {0:'song_0',1:'song_1',2:'song_2',3:'song_3',4:'song_4',5:'song_5',6:'song_6',7:'song_7',8:'song_8',9:'song_9',10:'song_10',11:'song_11',12:'song_12',
                          13:'song_13',14:'song_14',15:'song_15',16:'song_16',17:'song_17',18:'song_18',19:'song_19',20:'song_20',21:'song_21',22:'song_22',23:'song_23',
                          24:'song_24',25:'song_25',26:'song_26',27:'song_27',28:'song_28',29:'song_29',30:'song_30',31:'song_31'},inplace=True)
X_songembed



## tag 임배딩
X = X_songembed
tags = [p for p in X['tags'] if len(p) != 1]

m = Word2Vec(tags, size=32)
tag_vector = m.wv

tags = tag_vector.vocab.keys()
tag_vector_lst = [tag_vector[v] for v in tags]
from gensim.models import KeyedVectors
tag_vector.save_word2vec_format('tag2v_over50')


# inference 
tag_vector = gensim.models.KeyedVectors.load_word2vec_format('tag2v_over50')

def tag_embed(x):
    tem = []
    for tag in x:
        try:
            tem.append(tag_vector.get_vector(tag))
        except KeyError as e:
            pass
    if tem == []:
        return np.zeros(32)
    else:
        return np.mean(tem,axis=0)

X_total = pd.concat([X_songembed ,pd.DataFrame(list(X['tags'].apply(tag_embed)))],axis=1)
X_total.rename(columns = {0:'tag_0',1:'tag_1',2:'tag_2',3:'tag_3',4:'tag_4',5:'tag_5',6:'tag_6',7:'tag_7',8:'tag_8',9:'tag_9',10:'tag_10',11:'tag_11',12:'tag_12',
                          13:'tag_13',14:'tag_14',15:'tag_15',17:'tag_17',18:'tag_18',19:'tag_19',20:'tag_20',21:'tag_21',22:'tag_22',23:'tag_23',
                          24:'tag_24',25:'tag_25',26:'tag_26',27:'tag_27',28:'tag_28',29:'tag_29',30:'tag_30',31:'tag_31'},inplace=True)

del X_total['tags']
del X_total['song_gn_gnr_basket_flatten']
del X_total['songs']

X_total


## Season, year
train = X_total
train['gn_updt_date'] = pd.to_datetime(train['gn_updt_date'], format='%Y-%m-%d %H:%M:%S', errors='raise')

train['date'] = train['gn_updt_date'].dt.date         # YYYY-MM-DD(문자)
train['year']  = train['gn_updt_date'].dt.year         # 연(4자리숫자)
train['month']  = train['gn_updt_date'].dt.month        # 월(숫자)
train['season'] = train['gn_updt_date'].dt.quarter

train['season'][train['month'].isin([1,2,12])] = 4  # 겨울
train['season'][train['month'].isin([3,4,5])] = 1   # 봄
train['season'][train['month'].isin([6,7,8])] = 2  # 여름
train['season'][train['month'].isin([9,10,11])] = 3  # 가을

df = pd.DataFrame(columns=['year_section'])
train = pd.concat([train,df])

train['year_section'][train['year'].isin([2005,2006,2007,2008,2009,2010,2011,2012,2013])] = 1
train['year_section'][train['year'].isin([2014,2015])] = 2
train['year_section'][train['year'].isin([2016])] = 3
train['year_section'][train['year'].isin([2017])] = 4
train['year_section'][train['year'].isin([2018])] = 5
train['year_section'][train['year'].isin([2019,2020])] = 6

del train['date']
del train['year']
del train['month']
del train['gn_updt_date']


## scaling and one-hot encoding
X_train = train
X_train['artist_id_basket_count'] = (X_train['artist_id_basket_count'] - X_train['artist_id_basket_count'].mean())/X_train['artist_id_basket_count'].std()
X_train['song_gn_gnr_basket_count'] = (X_train['song_gn_gnr_basket_count'] -X_train['song_gn_gnr_basket_count'].mean())/X_train['song_gn_gnr_basket_count'].std()

df_season = pd.get_dummies(X_train['season']).add_prefix('season') 
df_year = pd.get_dummies(X_train['year_section']).add_prefix('year_section') 
X_train = pd.concat([X_train,df_season,df_year],axis=1)

del X_train['season']
del X_train['year_section']

X_train = X_train.rename(columns={'season1.0':'season_1','season2.0':'season_2','season3.0':'season_3','season4.0':'season_4','year_section1.0':'year_1','year_section2.0':'year_2','year_section3.0':'year_3','year_section4.0':'year_4','year_section5.0':'year_5'})

X_train['id'] = X_train['id'].astype(int)
X_train = X_train.set_index('id')
X_train

X_train.to_json('last_t2r_X.json')


## make it sparse matrix
X_train = X_train.astype(float)
X_spr = spr.csr_matrix(X_train)
X_spr

train_songs_A

full = hstack((X_spr,train_songs_A))
full = full.tocsc() ; full

mmwrite('rerank_song_last.mtx', full)



