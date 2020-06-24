# -*- coding: utf-8 -*-
"""FM_train

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lC_kZrYkki4EIUftLyxC7o1oq-jF57Y8

[카카오 아레나 포럼
_CF](https://arena.kakao.com/forum/topics/227)
"""

from google.colab import drive
drive.mount('/content/drive')

!pip install fire

import numpy as np
import pandas as pd

import scipy.sparse as spr
import pickle

#song_meta = pd.read_json("drive/My Drive/KAKAO/song_meta.json")

cd /content/drive/My Drive/Kakao arena

"""# Pre-Processing
min_cnt >= 5인 곡만 train set에 남기기
"""

train = pd.read_json('data/meta/train.json', typ = 'frame')

train.shape

# 플레이리스트 아이디(id)와 수록곡(songs) 추출
plylst_song_map = train[['id', 'songs']]

# unnest songs
plylst_song_map_unnest = np.dstack(
    (
        np.repeat(plylst_song_map.id.values, list(map(len, plylst_song_map.songs))), 
        np.concatenate(plylst_song_map.songs.values)
    )
)

# unnested 데이터프레임 생성 : plylst_song_map
plylst_song_map = pd.DataFrame(data = plylst_song_map_unnest[0], columns = plylst_song_map.columns)
plylst_song_map['id'] = plylst_song_map['id'].astype(str)
plylst_song_map['songs'] = plylst_song_map['songs'].astype(str)

# unnest 객체 제거
del plylst_song_map_unnest

value = plylst_song_map['songs'].value_counts() > 5

df = plylst_song_map['songs'].value_counts().rename_axis('unique_values').to_frame('counts')
df = df.reset_index() ; df
over_5 = df[df['counts']>=5]

requred_song = over_5['unique_values'].tolist()
requred_song = set(list(map(int, requred_song)))

df = pd.DataFrame(columns=['over5_songs'])
train = pd.concat([train,df])

for i in range(train.shape[0]):     
    result = set(list(map(int, train['songs'][i]))) & requred_song
    train['over5_songs'][i] = list(result)

del train['songs']
train.rename(columns = {'over5_songs' : 'songs'}, inplace = True)
train.head()

train.to_json('train_over5.json', orient='table')

# def write_json(data, fname):
#     def _conv(o):
#         if isinstance(o, (np.int64, np.int32)):
#             return int(o)
#         raise TypeError

#     parent = os.path.dirname(fname)
#     distutils.dir_util.mkpath("./arena_data/" + parent)
#     with io.open("./arena_data/" + fname, "w", encoding="utf-8") as f:
#         json_str = json.dumps(data, ensure_ascii=False, default=_conv)
#         f.write(json_str)

"""train/test로 split -> score modeling

split_data.py가 안되서 그냥 train_test split 씀
"""

# !python split_data.py run res/train_1.json

# train = pd.read_json("arena_data/orig/train.json")
# test = pd.read_json("arena_data/questions/val.json")

train = pd.read_json('train_over5.json',orient='table')

train.head()

from sklearn.model_selection import train_test_split

train, test= train_test_split(train, test_size=0.2,shuffle=True, random_state=1004)

"""# Util function"""

import io
import os
import json
import distutils.dir_util
from collections import Counter

import numpy as np

def write_json(data, fname):
    def _conv(o):
        if isinstance(o, np.int64) or isinstance(o, np.int32):
            return int(o)
        raise TypeError

    parent = os.path.dirname(fname)
    distutils.dir_util.mkpath("./arena_data/" + parent)
    with io.open("./arena_data/" + fname, "w", encoding="utf8") as f:
        json_str = json.dumps(data, ensure_ascii=False, default=_conv)
        f.write(json_str)


def load_json(fname):
    with open(fname, encoding='utf8') as f:
        json_obj = json.load(f)

    return json_obj


def debug_json(r):
    print(json.dumps(r, ensure_ascii=False, indent=4))

class CustomEvaluator:
    def _idcg(self, l):
        return sum((1.0 / np.log(i + 2) for i in range(l)))

    def __init__(self):
        self._idcgs = [self._idcg(i) for i in range(101)]

    def _ndcg(self, gt, rec):
        dcg = 0.0
        for i, r in enumerate(rec):
            if r in gt:
                dcg += 1.0 / np.log(i + 2)

        return dcg / self._idcgs[len(gt)] # self._idcgs[len(gt)] = 0

    def _eval(self, gt_fname, rec_fname):
        gt_playlists = load_json(gt_fname)
        gt_dict = {g["id"]: g for g in gt_playlists} # 정답 {플레이리스트 아이디 : 플레이리스트 정보 쭉}
        rec_playlists = load_json(rec_fname)

        gt_ids = set([g["id"] for g in gt_playlists]) # 정답 플레이리스트 아이디
        rec_ids = set([r["id"] for r in rec_playlists]) # 답안 플레이리스트 아이디 

        '''
        if gt_ids != rec_ids:
            print(f'gt_ids:{len(gt_ids)}')
            print(f'rec_ids:{len(rec_ids)}')
            raise Exception("결과의 플레이리스트 수가 올바르지 않습니다.")

        rec_song_counts = [len(p["songs"]) for p in rec_playlists] #답안 플레이리스트 song 개수 집합으로. (100 아닌거 있나 걸러내려고)
        rec_tag_counts = [len(p["tags"]) for p in rec_playlists]

        if set(rec_song_counts) != set([100]):
            print(f'rec_song_counts={set(rec_song_counts)}')
            raise Exception("추천 곡 결과의 개수가 맞지 않습니다.")

        if set(rec_tag_counts) != set([10]):
            raise Exception("추천 태그 결과의 개수가 맞지 않습니다.")

        rec_unique_song_counts = [len(set(p["songs"])) for p in rec_playlists]
        rec_unique_tag_counts = [len(set(p["tags"])) for p in rec_playlists]

        if set(rec_unique_song_counts) != set([100]):
            print(f'rec_unique_song_counts={set(rec_unique_song_counts)}')
            raise Exception("한 플레이리스트에 중복된 곡 추천은 허용되지 않습니다.")

        if set(rec_unique_tag_counts) != set([10]):
            print(f'rec_unique_tag_counts={set(rec_unique_tag_counts)}')
            raise Exception("한 플레이리스트에 중복된 태그 추천은 허용되지 않습니다.")
        '''

        music_ndcg = 0.0
        tag_ndcg = 0.0

        for rec in rec_playlists:
            gt = gt_dict[rec["id"]]
            music_ndcg += self._ndcg(gt["songs"], rec["songs"][:100])
            tag_ndcg += self._ndcg(gt["tags"], rec["tags"][:10])

        music_ndcg = music_ndcg / len(rec_playlists)
        tag_ndcg = tag_ndcg / len(rec_playlists)
        score = music_ndcg * 0.85 + tag_ndcg * 0.15

        return music_ndcg, tag_ndcg, score

    def evaluate(self, gt_fname, rec_fname): # gt > 정답, rec > 제출 답안
        try:
            music_ndcg, tag_ndcg, score = self._eval(gt_fname, rec_fname)
            print(f"Music nDCG: {music_ndcg:.6}")
            print(f"Tag nDCG: {tag_ndcg:.6}")
            print(f"Score: {score:.6}")
        except Exception as e:
            print(e)

"""# Make Spare Matrix(score)"""

train['istrain'] = 1
test['istrain'] = 0

n_train = len(train)
n_test = len(test)

# train + test
plylst = pd.concat([train,test], ignore_index=True)
plylst["nid"] = range(n_train+n_test)

plylst.head(5)

# id <-> nid
plylst_id_nid = dict(zip(plylst["id"],plylst["nid"]))
plylst_nid_id = dict(zip(plylst["nid"],plylst["id"]))

plylst_tag = plylst['tags'][0:3]
for tgs in plylst_tag:
  print(tgs)
  for tg in tgs:
    print(tg)

tag_counter = Counter([tg for tgs in plylst_tag for tg in tgs])
print(tag_counter)

print('*******************')

tag_dict = {x:tag_counter[x] for x in tag_counter}
print(tag_dict)

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

plylst_use = plylst[['istrain','nid','updt_date','songs_id','tags_id']]
plylst_use.loc[:,'num_songs'] = plylst_use['songs_id'].map(len)
plylst_use.loc[:,'num_tags'] = plylst_use['tags_id'].map(len)
plylst_use = plylst_use.set_index('nid')

plylst_use.head(5)

plylst_train = plylst_use.iloc[:n_train,:]
plylst_test = plylst_use.iloc[n_train:,:]

len(plylst_test)

np.random.seed(33)
n_sample = 23015

test = plylst_test.iloc[np.random.choice(range(n_test), size=n_sample,replace=False),:]

test.head(5)

import math
from tqdm import tqdm

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

train_songs_A  # 14만 곡으로 줄었음

train_tags_A # 태그는 그대로

## 전체 ply에 해당 곡이 몇개나 들어 있는지
from tqdm import tqdm

cnt = []
for i in tqdm(range(144674)):
    cnt.append(len(train_songs_A.T[i].data))

## target score 추가하기

for i in range(144674):
    df = pd.DataFrame(columns=['target_score_{}'.format(i)])
    train = pd.concat([train,df])     
    result = train_songs_A.T[i].T.todense()
    train['target_score_{}'.format(i)] = result

    if i % 100 == 0:
        print("{}/144674_{}%".format(i,i//1446))

"""# Modeling
input이 spare 하지 않다 -> 구지 FM을 쓸 필요가 없다.  
XGBoost나 다른 regression 고려
"""

plylst_train.head()

train = plylst_train

"""## Feature engineering

### Season 
봄, 여름, 가을, 겨울
"""

train['updt_date'] = pd.to_datetime(train['updt_date'], format='%Y-%m-%d %H:%M:%S', errors='raise')

train['date'] = train['updt_date'].dt.date         # YYYY-MM-DD(문자)
train['year']     = train['updt_date'].dt.year         # 연(4자리숫자)
train['month']      = train['updt_date'].dt.month        # 월(숫자)
train['season'] = train['updt_date'].dt.quarter

train['season'][train['month'].isin([1,2,12])] = 4  # 겨울
train['season'][train['month'].isin([3,4,5])] = 1   # 봄
train['season'][train['month'].isin([6,7,8])] = 2  # 여름
train['season'][train['month'].isin([9,10,11])] = 3  # 가을

train.head(3)

"""### Year_section"""

df = pd.DataFrame(columns=['year_section'])
train = pd.concat([train,df])

train['year_section'][train['year'].isin([2005,2006,2007,2008,2009,2010,2011,2012])] = 1
train['year_section'][train['year'].isin([2013,2014])] = 2
train['year_section'][train['year'].isin([2015,2016])] = 3
train['year_section'][train['year'].isin([2017,2018])] = 4
train['year_section'][train['year'].isin([2019,2020])] = 5

del train['date']
del train['updt_date']
del train['istrain']

train.head(3)

"""기존 month, year은 공선성 문제가 있을테니 제거??"""

from xgboost import plot_importance
from xgboost import XGBRegressor

"""# CF inference"""

from tqdm import tqdm

def rec(pids):
  tt = 1

  res = []

  for pid in tqdm(pids): # pids로 test.index 들어옴. 따라서 플레이리스트마다 아래 과정 실행하는 것.
    p = np.zeros((n_songs,1)) 
    # n_songs) 플레이리스트에 있는 곡만 포함한듯?
    # song 개수만큼 0으로 채워진 array 생성
    p[test.loc[pid,'songs_id']] = 1 # 이번 플레이리스트에서 있는 song_id는 1로 바꿔줌

    val = train_songs_A.dot(p) 
    # train_songs_A > plylst-songs sparse 행렬 (115071 x 638336) / p > (638336 x 1)
    # 전체 플레이리스트 정보 x 이번 플레이리스트 > 둘 다 가지고 있는 곡 개수

    songs_already = test.loc[pid,"songs_id"]
    tags_already = test.loc[pid,"tags_id"]

    cand_song = train_songs_A_T.dot(val)
    # 플레이리스트 쌍마다 공통 곡의 수를 가중치로 하여 곡과 태그를 추천하는 방식
    # (638336 x 115071) x (115071 x 1) = (638336 x 1) > 어떤 플레이리스트에 있는 곡(값 1) x 겹치는 곡 수를 더해서 곡 별 일종의 점수 부여
    cand_song_idx = cand_song.reshape(-1).argsort()[-150:][::-1] # reshape) ex. (3,1) > (3,) / argsort) 작은 값부터 순서대로 데이터 인덱스 반환 / 큰 값부터 150개 자르고, 순서 뒤집음

    cand_song_idx = cand_song_idx[np.isin(cand_song_idx, songs_already) == False][:100]
    rec_song_idx = [song_sid_id[i] for i in cand_song_idx]

    cand_tag = train_tags_A_T.dot(val)
    cand_tag_idx = cand_tag.reshape(-1).argsort()[-15:][::-1]

    cand_tag_idx = cand_tag_idx[np.isin(cand_tag_idx, tags_already) == False][:10]
    rec_tag_idx = [tag_tid_id[i] for i in cand_tag_idx]

    res.append({
        "id":plylst_nid_id[pid],
        "songs":rec_song_idx,
        "tags":rec_tag_idx
    })

    if tt % 1000 == 0:
      print(tt)

    tt += 1
  return res

answers = rec(test.index)

write_json(answers,"results/results.json")

evaluator = CustomEvaluator()
evaluator.evaluate("arena_data/answers/val.json", "arena_data/results/results.json")

# evaluator 주석처리한 에러 잡아줘야함 > 100곡, 10개 채우게
# 점수 준 게 막 엄청난 성능향상을 가져오진 않는듯