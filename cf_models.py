import pandas as pd
import numpy as np
import math
import os
import json
import scipy.sparse as spr  ## sparse matrix 만드는 함수
from implicit.evaluation import  *
from implicit.als import AlternatingLeastSquares as ALS
from implicit.bpr import BayesianPersonalizedRanking as BPR
from collections import Counter
from sklearn.utils import shuffle
from tqdm import tqdm
from itertools import groupby
from arena_util import write_json
from evaluate import ArenaEvaluator

class CF:
    def __init__(self):
        self.train = pd.read_json("arena_data/orig/train.json", encoding='utf-8')
        self.val = pd.read_json("arena_data/questions/val.json", encoding='utf-8')
        self.ans = pd.read_json("arena_data/answers/val.json", encoding='utf-8')

        self.train['istrain'] = 1       # train이랑 val의 차이를 넣어놓는 이유 : nid 만들 때 순서대로 전부 새로운 id를 부여하기 때문
        self.val['istrain'] = 0         # 나중에 잘 나누려고~

        self.n_train = len(self.train)
        self.n_test = len(self.val)

        # train + test
        self.plylst = pd.concat([self.train, self.val], ignore_index=True)  # train이랑 test랑 합치기
        # playlist id
        self.plylst["nid"] = range(self.n_train + self.n_test)   # nid 값 부여하기~~

        # id <-> nid
        self.plylst_id_nid = dict(zip(self.plylst["id"],self.plylst["nid"]))   # nid랑 id 값 각 값이 어떤 값을 나타내는지 저장 dict으로 저장
        self.plylst_nid_id = dict(zip(self.plylst["nid"],self.plylst["id"]))   # 앞이 key값, 뒤가 value (id_nid는 id가 key 값, nid가 value 값)

        plylst_tag = self.plylst['tags']
        tag_counter = Counter([tg for tgs in plylst_tag for tg in tgs])  # 각 tag가 몇개 있는지 저장 dict을 () 값으로 묶은 Counter 객체
        tag_dict = {x: tag_counter[x] for x in tag_counter}             # 그래서 dict으로 풀어줘야함

        self.tag_id_tid = dict()
        self.tag_tid_id = dict()
        for i, t in enumerate(tag_dict):            # tag에는 tid 값 부여하기~
            self.tag_id_tid[t] = i
            self.tag_tid_id[i] = t

        self.n_tags = len(tag_dict)                  # n_tags에 tag값 부여하기

        plylst_song = self.plylst['songs']           # 각 plylst의 song들
        song_counter = Counter([sg for sgs in plylst_song for sg in sgs])
        song_dict = {x: song_counter[x] for x in song_counter}

        self.song_id_sid = dict()
        self.song_sid_id = dict()
        for i, t in enumerate(song_dict):       # song에는 sid 값 부여하기~
            self.song_id_sid[t] = i
            self.song_sid_id[i] = t

        self.n_songs = len(song_dict)

        # plylst가 가지고 있는 songs랑 tags들을 전부 sid, tid로 태깅하기
        self.plylst['songs_id'] = self.plylst['songs'].map(lambda x: [self.song_id_sid.get(s) for s in x if self.song_id_sid.get(s) != None])
        self.plylst['tags_id'] = self.plylst['tags'].map(lambda x: [self.tag_id_tid.get(t) for t in x if self.tag_id_tid.get(t) != None])

        # 이제 사용할 것들만 남기기!! istrain 이랑 nid sid tid 그리고 date까지만 사용
        plylst_use = self.plylst[['istrain','nid','updt_date','songs_id','tags_id']]
        plylst_use.loc[:,'num_songs'] = plylst_use['songs_id'].map(len) # 해당 plylst의 song 개수 몇개인지
        plylst_use.loc[:,'num_tags'] = plylst_use['tags_id'].map(len)  # 해당 plylst의 map 개수 몇개인지
        plylst_use = plylst_use.set_index('nid')    # index를 nid로 변경

        # istrain column 가지고 합쳤던 train이랑 val 나누기
        self.plylst_train = plylst_use.iloc[:self.n_train,:]
        self.plylst_test = plylst_use.iloc[self.n_train:,:]       

    def __call__(self, mode, song_ntop = 500, tag_ntop = 50):
        train_songs_A, train_tags_A, test_songs_A, test_tags_A = self.mkspr(self.train, self.val)
        if mode == 'cf':
            res = self.cf_(train_songs_A, train_tags_A, 500, 50)
        elif mode == 'mf':
            res = self.mf_(train_songs_A, train_tags_A, test_songs_A, test_tags_A, 50, 500, 50)
        # elif mode == 'ncf':
        #     res = self.ncf_()
        else:
            print('cf, mf, ncf 중에 하나임')

        #self.eval_rate(res)
        #self.eval_dcg(res)

        return res

    def rating(self,num,mode):
        if mode =='train':
            constant = 8.66
        else: #mode =='test':
            constant = 7.66

        return [-math.log(x+1,2)+constant for x in range(num)]

    def mkspr(self, train, val):

        row = np.repeat(range(self.n_train), self.plylst_train['num_songs']) # range => id 라서 id를 songs 개수만큼 반복한게 row로 사용
        col = [song for songs in self.plylst_train['songs_id'] for song in songs]  # 모든 plyst의 songs를 순서대로 쭉~
        dat_series = self.plylst_train['num_songs'].map(lambda x: self.rating(x,'train'))
        dat = [y for x in dat_series for y in x] 
        train_songs_A = spr.csr_matrix((dat, (row, col)), shape=(self.n_train, self.n_songs)) # matrix 생성

        row = np.repeat(range(self.n_train), self.plylst_train['num_tags'])
        col = [tag for tags in self.plylst_train['tags_id'] for tag in tags]
        dat = np.repeat(1, self.plylst_train['num_tags'].sum())
        train_tags_A = spr.csr_matrix((dat, (row, col)), shape=(self.n_train, self.n_tags))

        row = np.repeat(range(self.n_test), self.plylst_test['num_songs']) # range => id 라서 id를 songs 개수만큼 반복한게 row로 사용
        col = [song for songs in self.plylst_test['songs_id'] for song in songs]  # 모든 plyst의 songs를 순서대로 쭉~
        dat_series = self.plylst_test['num_songs'].map(lambda x: self.rating(x,'test'))
        dat = [y for x in dat_series for y in x] 
        test_songs_A = spr.csr_matrix((dat, (row, col)), shape=(self.n_test, self.n_songs))

        row = np.repeat(range(self.n_test), self.plylst_test['num_tags'])
        col = [tag for tags in self.plylst_test['tags_id'] for tag in tags]
        dat = np.repeat(1, self.plylst_test['num_tags'].sum())
        test_tags_A = spr.csr_matrix((dat, (row, col)), shape=(self.n_test, self.n_tags))

        return train_songs_A, train_tags_A, test_songs_A, test_tags_A

    def cf_(self, train_songs_A, train_tags_A,test_songs_A,test_tags_A, song_ntop = 500, tag_ntop = 50):

        train_songs_A_T = train_songs_A.T.tocsr() # shape) n_songs * n_train ply
        train_tags_A_T = train_tags_A.T.tocsr() # shape) n_tags * n_train ply

        res = []

        song_val = test_songs_A.dot(train_songs_A_T) # n_test * n_train
        tag_val = test_tags_A.dot(train_tags_A_T)

        cand_song_matrix = song_val.dot(train_songs_A) # n_test * n_songs
        cand_tag_matrix = tag_val.dot(train_tags_A)

        del song_val
        del tag_val

        for r,pid in tqdm(enumerate(self.plylst_test.index),0):

            songs_already = self.plylst_test.loc[pid, "songs_id"]
            tags_already = self.plylst_test.loc[pid, "tags_id"] 

            song_row = cand_song_matrix.getrow(r).toarray().reshape(-1,) # 1 * n_songs > 점수 행렬 
            cand_song_idx = song_row.argsort()[-song_ntop-50:][::-1] # 점수 순 idx(= 곡 sid) sort

            cand_song_idx = cand_song_idx[np.isin(cand_song_idx, songs_already) == False][:song_ntop] # 원래 있던 곡 제외, 상위 n개
            rec_song_score = [song_row[i] for i in cand_song_idx]

            tag_row = cand_tag_matrix.getrow(r).toarray().reshape(-1,) 
            cand_tag_idx = tag_row.argsort()[-tag_ntop-5:][::-1]

            cand_tag_idx = cand_tag_idx[np.isin(cand_tag_idx, tags_already) == False][:tag_ntop]
            rec_tag_score = [tag_row.data[i] for i in cand_tag_idx]

            res.append({
                        "id": self.plylst_nid_id[pid],
                        "songs": [self.song_sid_id[i] for i in cand_song_idx], #sid
                        "tags": [self.tag_tid_id[i] for i in cand_tag_idx],  #tid
                        "songs_score":rec_song_score,
                        "tags_score":rec_tag_score
                    })
            
        return res 
        # test(23000여개) > 약 6분 소요. song) 30.2% , tag) 29.3% (바꾼 기준 하에서)     

    def mf_(self, train_songs_A, train_tags_A, test_songs_A, test_tags_A, iteration, song_ntop = 500, tag_ntop = 50):
        
        print(f'iters:{iteration}')
        res = []

        als_model = ALS(factors=1024, regularization=0.08, iterations=iteration,use_gpu=True) # 16, 1024
        als_model.fit(train_songs_A.T)   

        als_model_tag = ALS(factors=1024, regularization=0.08, iterations=iteration ,use_gpu=True)
        als_model_tag.fit(train_tags_A.T)

        for pid in tqdm(range(self.n_test)):  ## 한 15분 정도 걸림
            song_rec = als_model.recommend(pid, test_songs_A, N=song_ntop)  # N 이 몇 개 추천받을지
            tag_rec = als_model_tag.recommend(pid, test_tags_A, N=tag_ntop)  # N 이 몇 개 추천받을지
 
            # 튜플 곡, 점수로 분리
            rec_song_idx = [self.song_sid_id[x[0]] for x in song_rec]
            rec_song_score = [x[1] for x in song_rec]
            rec_tag_idx = [self.tag_tid_id[x[0]] for x in tag_rec]
            rec_tag_score = [x[1] for x in tag_rec]

            res.append({
                        "id": self.plylst_nid_id[pid+self.n_train],
                        "songs": rec_song_idx, 
                        "tags": rec_tag_idx,
                        "songs_score":rec_song_score,
                        "tags_score":rec_tag_score
                    })

        return res

    def eval_rate(self, res):
        predict = pd.DataFrame(res)[['id','songs','tags']]
        answer = self.ans[['id', 'songs', 'tags']]
      
        # compare = pd.merge(predict,answer,how='left',on='id') > 에러..
        predict[['songs_ans','tags_ans']] = answer[['songs','tags']]

        ''' 
        원래 eval_rate 
        predict['correct_songs']= predict.apply(lambda x:len(set(x['songs'])&set(x['songs_ans']))/len(x['songs_ans']) * 100, axis=1)
        predict['correct_tags'] = predict.apply(lambda x:len(set(x['tags'])&set(x['tags_ans']))/len(x['tags_ans']) * 100, axis=1)
        '''

        predict['correct_songs']= predict.apply(lambda x:len(set(x['songs'])&set(x['songs_ans'])))
        predict['correct_tags'] = predict.apply(lambda x:len(set(x['tags'])&set(x['tags_ans'])))

        ans_songs_num = predict['songs_ans'].map(len).sum()
        ans_tags_num = predict['tags_ans'].map(len).sum()

        correct_songs = predict.loc[:,"correct_songs"].sum()
        correct_tags = predict.loc[:,"correct_tags"].sum()

        print(f"Songs: {(correct_songs/ans_songs_num) * 100:.3}%")
        print(f"Tags: {(correct_tags/ans_tags_num) * 100 :.3}%")


    def eval_dcg(self,res):
        write_json(res,"results/results.json")
        evaluator = ArenaEvaluator()
        evaluator.evaluate(self.ans, "arena_data/results/results.json")


if __name__ == "__main__":

    a = CF()
    res = a(mode = 'mf')

    # def ncf_(self, train_songs_A, train_tags_A, song_ntop = 500, tag_ntop = 50):
