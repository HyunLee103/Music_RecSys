from sklearn.externals import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb


"""
x : input matrix , (n X # of feature)
ids : 500개씩 추천 된 곡 id 리스트  , (n X 500)

"""

def song_rerank(x, ids):
    totol = []
    for idx in x:
        ply = x[idx]
        sid = ids[idx]
        res =[]
        for i in sid:
            lgbm = joblib.load('song_regression_2/{}_model'.format(i))
            res.append(lgbm.predict(x))
        total.append(res)
    return total

    

"""
x : x : input matrix , (n X # of feature)
tag : 100개씩 추천 된 태그 리스트 (n X 100)

"""

def tag_rerank(x, tag):
    totol = []
    for idx in range(len(x)):
        ply = x[idx]
        tags = tag[idx]
        res =[]
        for i in tags:
            lgbm = joblib.load('song_regression_2/{}_model'.format(i))
            res.append(lgbm.predict(x))
        total.append(res)
    return total