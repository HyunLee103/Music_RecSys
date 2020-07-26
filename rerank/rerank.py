from sklearn.externals import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm
"""
unique song 225495개
unique tag 3089개
"""
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

val = load_json('val_results.json')
val = pd.DataFrame(val)

X = load_json('rerank_val_X')
X = pd.DataFrame(X)



val_t2r = load_json('val_t2r.json')
val_t2r = pd.DataFrame(val_t2r)


plylst_song_map = val[['id', 'songs']]

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

unique_song = plylst_song_map['songs'].unique()
len(unique_song)

tag_x = load_json('rerank_val_X')
tag_x = pd.DataFrame(tag_x)
tag_x

## feature importance 시각화
# lgbm_1 = joblib.load('tag_regression_3/{}_model'.format('겨울'))
# lgbm_2 = joblib.load('tag_regression_3/{}_model'.format('기분전환'))
# lgbm_1.feature_importances_
# lgbm_2.feature_importances_


# feature_imp = pd.DataFrame(sorted(zip(lgbm_1.feature_importances_,tag_x.columns)), columns=['Value','Feature'])

# plt.figure(figsize=(20, 10))
# sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
# plt.title('LightGBM Features (avg over folds)')
# plt.tight_layout()
# plt.show()


def tag_regression(unique,x): 
    res = {}
    for sid in tqdm(unique):
        try:
            lgbm = joblib.load('tag_regression_3/{}_model'.format(sid))
            res['{}'.format(sid)] = lgbm.predict(x)
        except:
            pass
    return res

tag_weigt = tag_regression(unique_song,tag_x)





train = pd.read_json('data/meta/train.json',typ = 'frame')
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

value = plylst_song_map['songs'].value_counts() > 150

df = plylst_song_map['songs'].value_counts().rename_axis('unique_values').to_frame('counts')
df = df.reset_index() ; df.head(3)
over_5 = df[df['counts']>=150]


requred_song = over_5['unique_values'].tolist()
len(requred_song)


X = load_json('last_t2r_X.json')
X = pd.DataFrame(X)
X = X.reset_index()

X['index'] = X['index'].astype(int)
X = X.sort_values(by='index')

def song_regression(unique,x): 
    res = {}
    for sid in tqdm(unique):
        try:
            lgbm = joblib.load('song_regression_last/{}_model'.format(sid))
            res['{}'.format(sid)] = lgbm.predict(x)
        except:
            pass
    return res

song_weight = song_regression(requred_song,X)



val = load_json('val_results.json')
val = pd.DataFrame(val)

val = val.sort_values(by='id')
tem = val

tags = tem['songs']
scores = tem['songs_score']

def normalize(x):
    return ((x - x.min())/(x.max()-x.min()))

def tag_rerank(): 
    df_2 = pd.DataFrame(index=range(0,23015),columns=['new_songs'])
    for i in tqdm(range(23015)):

        a = np.zeros(len(scores.iloc[i]))
        for j,tag in enumerate(tags.iloc[i]):
            try:
                a[j] = (song_weight['{}'.format(tag)[i]])
            except:
                # print('no_tag') ## 약 30개
                continue
         
        try:

            k  = (normalize(a)) * np.array(scores.iloc[i])
            # s = np.concatenate([scores.iloc[i][:20],k,scores.iloc[i]])
            df = pd.concat([pd.DataFrame(k,columns=['score']),pd.DataFrame(tags.iloc[i],columns=['songs'])],axis=1)
            # new_tags = df.sort_values(by='score',ascending=False).iloc[0:10]['tags'].tolist()
            df_2['new_songs'][i] = (df.sort_values(by='score',ascending=False).iloc[0:100]['songs'].tolist())
            
        except:
            print('array_size_error') ## 약 50개
 
    return df_2
    
reranked_song = tag_rerank()

val = val.reset_index()
del val['songs']
reranked_tag_2 = pd.concat([val,reranked_song],axis=1)

reranked_tag_2 = reranked_tag_2.rename(columns={'new_songs':'songs'})

reranked_tag_2['songs'] = reranked_tag_2['songs'].map(lambda x : x[:100])
reranked_tag_2['tags'] = reranked_tag_2['tags'].map(lambda x : x[:10])

del reranked_tag_2['index']
del reranked_tag_2['songs_score']
del reranked_tag_2['tags_score']

reranked_tag_2.to_json('results.json',orient='records')
