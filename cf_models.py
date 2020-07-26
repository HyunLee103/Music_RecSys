import pandas as pd
import numpy as np
import math
import os
import json
import scipy.sparse as spr
from implicit.als import AlternatingLeastSquares as ALS
from collections import Counter
from tqdm import tqdm
from arena_util import write_json,remove_seen
from evaluate import ArenaEvaluator

class CF:
    def __init__(self):
        # json 파일 저장 없이 넘겨 받을 때는 init에 해당 변수 추가
        self.train = pd.read_json("arena_data/orig/train.json", encoding='utf-8')
        self.test = pd.read_json("arena_data/questions/val_hye.json", encoding='utf-8') #  곡/태그 추가된 상태
        # self.orig_test= pd.read_json("arena_data/orig/val.json", encoding='utf-8')
        self.ans = pd.read_json("arena_data/answers/val_hye.json", encoding='utf-8')

        # self.test_info = pd.read_json(val_ply)

        self.train['istrain'] = 1
        self.test['istrain'] = 0

        self.n_train = len(self.train)
        self.n_test = len(self.test)

        # val + train
        self.plylst = pd.concat([self.train,self.test])
        self.plylst["nid"] = range(self.n_train + self.n_test)

        self.plylst_id_nid = dict(zip(self.plylst["id"],self.plylst["nid"])) 
        self.plylst_nid_id = dict(zip(self.plylst["nid"],self.plylst["id"]))  

        plylst_song = self.plylst['songs']       
        song_counter = Counter([sg for sgs in plylst_song for sg in sgs])
        song_dict = {x: song_counter[x] for x in song_counter}

        self.song_id_sid = dict()
        self.song_sid_id = dict()
        for i, t in enumerate(song_dict): 
            self.song_id_sid[t] = i
            self.song_sid_id[i] = t

        self.n_songs = len(song_dict)

        plylst_tag = self.plylst['tags']
        tag_counter = Counter([tg for tgs in plylst_tag for tg in tgs])
        tag_dict = {x: tag_counter[x] for x in tag_counter} 

        self.tag_id_tid = dict()
        self.tag_tid_id = dict()
        for i, t in enumerate(tag_dict): 
            self.tag_id_tid[t] = i
            self.tag_tid_id[i] = t

        self.n_tags = len(tag_dict)

        self.plylst['songs_id'] = self.plylst['songs'].map(lambda x: [self.song_id_sid.get(s) for s in x if self.song_id_sid.get(s) != None])
        self.plylst['tags_id'] = self.plylst['tags'].map(lambda x: [self.tag_id_tid.get(t) for t in x if self.tag_id_tid.get(t) != None])

        # 이제 사용할 것들만 남기기!! istrain 이랑 nid sid tid 그리고 date까지만 사용
        plylst_use = self.plylst[['istrain','nid','updt_date','songs_id','tags_id']] # + 곡 추가 여부 받는 열//
        plylst_use.loc[:,'num_songs'] = plylst_use['songs_id'].map(len) # 해당 plylst의 song 개수 몇개인지
        plylst_use.loc[:,'num_tags'] = plylst_use['tags_id'].map(len)  # 해당 plylst의 map 개수 몇개인지
        plylst_use = plylst_use.set_index('nid') 

        # istrain column 가지고 합쳤던 train이랑 val 나누기
        self.plylst_train = plylst_use.iloc[:self.n_train,:]
        self.plylst_test = plylst_use.iloc[self.n_train:,:]

    def __call__(self, mode, song_ntop = 500, tag_ntop = 50):
        train_songs_A, train_tags_A, test_songs_A, test_tags_A = self.mkspr(self.train, self.test)
        if mode == 'cf':
            res = self.cf_(train_songs_A, train_tags_A,test_songs_A, test_tags_A)
            
        elif mode == 'mf':
            res = self.mf_(train_songs_A, train_tags_A,test_songs_A, test_tags_A,25)

        self.eval_rate(res)
        
        self.eval_dcg(res)

        return res

    def rating_song(self,num,mode):
        if mode =='train':
            constant = 8.66
        else: #mode =='test':
            constant = 7.66

        return [-math.log(x+1,2)+constant for x in range(num)]

    def rating_tag(self,num,mode):
        if mode =='train':
            constant = 4.6
        else: #mode =='test':
            constant = 3.9

        return [-math.log(x+1,2)+constant for x in range(num)]

    def mkspr(self,train,test):
        print("Making sparse matrix...")
        row = np.repeat(range(self.n_train), self.plylst_train['num_songs']) # range => id 라서 id를 songs 개수만큼 반복한게 row로 사용
        col = [song for songs in self.plylst_train['songs_id'] for song in songs]  # 모든 plyst의 songs를 순서대로 쭉~
        dat_series = self.plylst_train['num_songs'].map(lambda x: self.rating_song(x,'train'))
        dat = [y for x in dat_series for y in x]
        train_songs_A = spr.csr_matrix((dat, (row, col)), shape=(self.n_train, self.n_songs)) # matrix 생성

        row = np.repeat(range(self.n_train), self.plylst_train['num_tags'])
        col = [tag for tags in self.plylst_train['tags_id'] for tag in tags]
        dat_series = self.plylst_train['num_tags'].map(lambda x: self.rating_tag(x,'train'))
        dat = [y for x in dat_series for y in x]
        train_tags_A = spr.csr_matrix((dat, (row, col)), shape=(self.n_train, self.n_tags))

        row = np.repeat(range(self.n_test), self.plylst_test['num_songs']) # range => id 라서 id를 songs 개수만큼 반복한게 row로 사용
        col = [song for songs in self.plylst_test['songs_id'] for song in songs]  # 모든 plyst의 songs를 순서대로 쭉~
        dat_series = self.plylst_test['num_songs'].map(lambda x: self.rating_song(x,'test'))
        dat = [y for x in dat_series for y in x]
        test_songs_A = spr.csr_matrix((dat, (row, col)), shape=(self.n_test, self.n_songs))

        row = np.repeat(range(self.n_test), self.plylst_test['num_tags'])
        col = [tag for tags in self.plylst_test['tags_id'] for tag in tags]
        dat_series = self.plylst_test['num_tags'].map(lambda x: self.rating_tag(x,'test'))
        dat = [y for x in dat_series for y in x]
        test_tags_A = spr.csr_matrix((dat, (row, col)), shape=(self.n_test, self.n_tags))

        print("DONE")

        return train_songs_A, train_tags_A, test_songs_A, test_tags_A
    
    def cf_(self, train_songs_A, train_tags_A, test_songs_A, test_tags_A, song_ntop = 500, tag_ntop = 50):

        print("CF...")
        
        train_songs_A_T = train_songs_A.T.tocsr() # shape) n_songs * n_train ply
        train_tags_A_T = train_tags_A.T.tocsr() # shape) n_tags * n_train ply

        res = []

        song_val = test_songs_A.dot(train_songs_A_T)
        tag_val = test_tags_A.dot(train_tags_A_T)

        cand_song_matrix = song_val.dot(train_songs_A)
        cand_tag_matrix = tag_val.dot(train_tags_A)

        del song_val
        del tag_val

        for r,pid in tqdm(enumerate(self.plylst_test.index),0):

            songs_already = self.plylst_test.loc[pid, "songs_id"]
            tags_already = self.plylst_test.loc[pid, "tags_id"]

            '''
            if self.plylst_test.loc[pid,"song_added"]:
                songs_already = self.orig_test.loc[self.plylst_nid_id[pid],"songs"]
            
            if self.plylst_test.loc[pid,"tag_added"]:
                tags_already = self.orig_test.loc[self.plylst_nid_id[pid],"tags"]
            
            '''

            song_row = cand_song_matrix.getrow(r).toarray().reshape(-1,) # 1 * n_songs > 점수 행렬 
            cand_song_idx = song_row.argsort()[-song_ntop-50:][::-1] # 점수 순 idx(= 곡 sid) sort
            cand_song_idx = remove_seen(songs_already,cand_song_idx)[:song_ntop] # cand_song_idx에 있는 곡들 중 songs_already에 없는 곡
            #cand_song_idx = cand_song_idx[np.isin(cand_song_idx, songs_already) == False][:song_ntop] # 원래 있던 곡 제외, 상위 n개
            rec_song_score = [song_row[i] for i in cand_song_idx]

            tag_row = cand_tag_matrix.getrow(r).toarray().reshape(-1,) 
            cand_tag_idx = tag_row.argsort()[-tag_ntop-5:][::-1]
            cand_tag_idx = remove_seen(tags_already,cand_song_idx)[:tag_ntop]
            #cand_tag_idx = cand_tag_idx[np.isin(cand_tag_idx, tags_already) == False][:tag_ntop]
            rec_tag_score = [tag_row.data[i] for i in cand_tag_idx]

            res.append({
                        "id": self.plylst_nid_id[pid], # id로 반환
                        "songs": [self.song_sid_id[i] for i in cand_song_idx], # id로 반환
                        "tags": [self.tag_tid_id[i] for i in cand_tag_idx],  # id로 반환
                        "songs_score":rec_song_score,
                        "tags_score":rec_tag_score
                    })

        print("DONE")
            
        return res

    def mf_(self, train_songs_A, train_tags_A, test_songs_A, test_tags_A, song_ntop = 500, tag_ntop = 50, iteration=20):
        
        print(f'MF... iters:{iteration}')
        # 0711 기준 최고 하이퍼파라미터) * 100, song - 256, tag - 32, reg = 0.1, epoch 20 > song 56.4%, tag 61.3%
        
        res = []

        songs_A = spr.vstack([test_songs_A,train_songs_A])
        tags_A = spr.vstack([test_tags_A,train_tags_A])

        als_model = ALS(factors=256, regularization=0.08, use_gpu=True, iterations=iteration) # epoch
        als_model.fit(songs_A.T * 100)   

        als_model_tag = ALS(factors=32, regularization=0.08, use_gpu=True, iterations=iteration)
        als_model_tag.fit(tags_A.T * 100)

        #rec_song = als_model.recommend_all(train_songs_A,N=500) 
        #rec_tag = als_model_tag.recommend_all(train_tags_A,N=50) # list (no score)

        for pid in tqdm(range(test_songs_A.shape[0])):
        
            if self.plylst_test.loc[(self.n_train+pid),"song_dirty"] == 1:
                cand_song = als_model.recommend(pid, test_songs_A, N=song_ntop+50, filter_already_liked_items=False)


            else:
                cand_song = als_model.recommend(pid, test_songs_A, N=song_ntop, filter_already_liked_items=True)

            if self.plylst_test.loc[(self.n_train+pid),"tag_dirty"] == 1:
                cand_tag = als_model_tag.recommend(pid, test_tags_A, N=tag_ntop+5,  filter_already_liked_items=True)
                #tags_already = self.orig_test[self.orig_test['id']== self.plylst_nid_id[self.n_train + pid]]['tags']
                #cand_tag = remove_seen(tags_already,cand_tag)[:tag_ntop]

            else:
                cand_tag = als_model_tag.recommend(pid, test_tags_A, N=tag_ntop,  filter_already_liked_items=True)

            rec_song_idx = [self.song_sid_id.get(x[0]) for x in cand_song]
            rec_song_score = [x[1] for x in cand_song]
            rec_tag_idx = [self.tag_tid_id.get(x[0]) for x in cand_tag]
            rec_tag_score = [x[1] for x in cand_tag]

            res.append({
                        "id": self.plylst_nid_id[self.n_train + pid],
                        "songs": rec_song_idx, 
                        "tags": rec_tag_idx,
                        "songs_score":rec_song_score,
                        "tags_score":rec_tag_score
                    })

        print("DONE")

        return res
        
    def eval_rate(self, res):
        print("Caculating rate...")
        predict = pd.DataFrame(res)[['id','songs','tags']]
        answer = self.ans[['id', 'songs', 'tags']]
      
        compare = pd.merge(predict,answer,how='left',on='id')
        compare.columns = ['id','songs','tags','songs_ans','tags_ans']

        compare['correct_songs']= compare.apply(lambda x:len(set(x['songs'])&set(x['songs_ans'])), axis=1)
        compare['correct_tags'] = compare.apply(lambda x:len(set(x['tags'])&set(x['tags_ans'])), axis=1)

        ans_songs_num = compare['songs_ans'].map(len).sum()
        ans_tags_num = compare['tags_ans'].map(len).sum()

        correct_songs = compare.loc[:,"correct_songs"].sum()
        correct_tags = compare.loc[:,"correct_tags"].sum()

        print(f"Songs: {(correct_songs/ans_songs_num) * 100:.3}%")
        print(f"Tags: {(correct_tags/ans_tags_num) * 100 :.3}%")

    
    def eval_dcg(self,res):
        print("Caculating dcg...")
        write_json(res,"results/results.json")
        evaluator = ArenaEvaluator()
        evaluator.evaluate("arena_data/answers/val_hye.json", "arena_data/results/results.json")

if __name__ == "__main__":

    model = CF()
    res = model(mode = 'mf')