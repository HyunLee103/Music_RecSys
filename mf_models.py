import pandas as pd
import numpy as np
from math import log
import scipy.sparse as spr
from implicit.als import AlternatingLeastSquares as ALS
from collections import Counter
from tqdm import tqdm
from arena_util import write_json, remove_seen
from evaluate import ArenaEvaluator

class CF:
    def __init__(self):
        # json 파일 저장 없이 넘겨 받을 때는 init에 해당 변수 추가
        self.train = pd.read_json("res/train.json", encoding='utf-8')
        self.val = pd.read_json("res/val.json", encoding='utf-8') #  곡/태그 둘 다 빈 ply만 제외한 상태
        self.test = pd.read_json("res/test.json", encoding='utf-8') #  곡/태그 둘 다 빈 ply만 제외한 상태

        self.train['istrain'] = 1
        self.val['istrain'] = 0
        self.test['istrain'] = 0

        self.n_train = len(self.train)
        self.n_val = len(self.val)
        self.n_test = len(self.test)

        # train + (val+test)
        self.plylst = pd.concat([self.train,self.val,self.test])
        self.plylst["nid"] = range(self.n_train + self.n_val +self.n_test)

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
        plylst_use = self.plylst[['istrain','nid','updt_date','songs_id','tags_id']] 
        plylst_use.loc[:,'num_songs'] = plylst_use['songs_id'].map(len) # 해당 plylst의 song 개수 몇개인지
        plylst_use.loc[:,'num_tags'] = plylst_use['tags_id'].map(len)  # 해당 plylst의 map 개수 몇개인지
        plylst_use = plylst_use.set_index('nid') 

        # istrain column 가지고 합쳤던 train이랑 val 나누기
        self.plylst_train = plylst_use.iloc[:self.n_train,:]
        self.plylst_val = plylst_use.iloc[self.n_train:self.n_train + self.n_val,:]
        self.plylst_test = plylst_use.iloc[self.n_train + self.n_val:,:]
        
        
    def __call__(self, mode, X = 'None', song_ntop = 500, tag_ntop = 50):

        if mode == 'mf':
            train_songs_A, train_tags_A, val_songs_A, val_tags_A ,test_songs_A, test_tags_A = self.mkspr_for_mf(mode=X)
            val_song_res, val_tag_res, test_song_res, test_tag_res = self.mf_(train_songs_A, train_tags_A, val_songs_A, val_tags_A, test_songs_A, test_tags_A, 500,50,20, True)

            val = self.make_result(val_song_res, val_tag_res)
            test = self.make_result(test_song_res, test_tag_res)

            print("Saving results...")
            print(f'len of val: {len(val)}')
            print(f'len of test: {len(test)}')

            return val, test # df

        elif mode == 'multi_mf': # song-tag 붙여서
            train_songs_A, train_tags_A, val_songs_A, val_tags_A ,test_songs_A, test_tags_A = self.mkspr_for_mf(mode= X)
            val, test = self.multi_mf_(train_songs_A, train_tags_A,val_songs_A, val_tags_A, test_songs_A, test_tags_A, False, 500, 50, 100, False)

            print("Saving results...")
            print(f'len of val: {len(val)}')
            print(f'len of test: {len(test)}')

            return val, test # list of dict
        
        elif mode =='meta_mf':
            train_songs_A, train_tags_A, val_songs_A, val_tags_A ,test_songs_A, test_tags_A = self.mkspr_for_mf(mode = X)
            val,test = self.multi_mf_(train_songs_A,train_tags_A,val_songs_A, val_tags_A, test_songs_A, test_tags_A, True, 500, 50, 100, False)

            print("Saving results...")
            print(f'len of val: {len(val)}')
            print(f'len of test: {len(test)}')

            return val, test # list of dict     

    def rating_song(self, num, mode):
        if mode =='train':
            constant = 8.66
        else: #mode =='test':
            constant = 7.66

        return [-log(x+1,2)+constant for x in range(num)]

    def rating_tag(self, num, mode):
        if mode =='train':
            constant = 4.6
        else: #mode =='test':
            constant = 3.9

        return [-log(x+1,2)+constant for x in range(num)]

    def mkspr_for_mf(self, mode = "no_X"):

        # val ( except X )
        self.plylst_val_song = self.plylst_val[self.plylst_val['num_songs']> 0]
        self.n_val_song = len(self.plylst_val_song)

        self.plylst_val_tag = self.plylst_val[self.plylst_val['num_tags']> 0]
        self.n_val_tag = len(self.plylst_val_tag)

        # test ( except X )
        self.plylst_test_song = self.plylst_test[self.plylst_test['num_songs']> 0]
        self.n_test_song = len(self.plylst_test_song)

        self.plylst_test_tag = self.plylst_test[self.plylst_test['num_tags']> 0]
        self.n_test_tag = len(self.plylst_test_tag)


        print("Making sparse matrix...")
        def mkspr(sn_train = self.n_train, splylst_train = self.plylst_train, tn_train = self.n_train, tplylst_train = self.plylst_train, mode='train'):
            row = np.repeat(range(sn_train), splylst_train['num_songs']) # range => id 라서 id를 songs 개수만큼 반복한게 row로 사용
            col = [song for songs in splylst_train['songs_id'] for song in songs]  # 모든 plyst의 songs를 순서대로 쭉~
            dat_series = splylst_train['num_songs'].map(lambda x: self.rating_song(x,mode))
            dat = [y for x in dat_series for y in x]
            songs_A = spr.csr_matrix((dat, (row, col)), shape=(sn_train, self.n_songs)) # matrix 생성

            row = np.repeat(range(tn_train), tplylst_train['num_tags'])
            col = [tag for tags in tplylst_train['tags_id'] for tag in tags]
            dat_series = tplylst_train['num_tags'].map(lambda x: self.rating_tag(x,mode))
            dat = [y for x in dat_series for y in x]
            tags_A = spr.csr_matrix((dat, (row, col)), shape=(tn_train, self.n_tags))

            return songs_A, tags_A
        
        ## A : with all data ( same rows ), B : with no X data ( different rows )
        train_songs_A, train_tags_A = mkspr(sn_train = self.n_train, splylst_train = self.plylst_train, \
            tn_train = self.n_train, tplylst_train = self.plylst_train, mode='train')
        val_songs_A, val_tags_A = mkspr(sn_train = self.n_val, splylst_train = self.plylst_val, \
            tn_train = self.n_val, tplylst_train = self.plylst_val, mode='test')        
        test_songs_A, test_tags_A = mkspr(sn_train = self.n_test, splylst_train = self.plylst_test, \
            tn_train = self.n_test, tplylst_train = self.plylst_test, mode='test')
        val_songs_B, val_tags_B = mkspr(sn_train = self.n_val_song, splylst_train = self.plylst_val_song, \
            tn_train = self.n_val_tag, tplylst_train = self.plylst_val_tag, mode='test')
        test_songs_B, test_tags_B = mkspr(sn_train = self.n_test_song, splylst_train = self.plylst_test_song, \
            tn_train = self.n_test_tag, tplylst_train = self.plylst_test_tag, mode='test')     

        print("DONE")

        if mode == "no_X":
            return train_songs_A, train_tags_A, val_songs_B, val_tags_B, test_songs_B, test_tags_B
        else: return train_songs_A, train_tags_A, val_songs_A, val_tags_A, test_songs_A, test_tags_A

    def mkspr_for_meta(self):
        meta = pd.read_json("res/meta_add1.json", encoding='utf-8')
        meta['id'] = meta['id'].map(self.plylst_id_nid)
        meta = meta.sort_values(by=['id'],axis=0)

        del meta['id']

        csr_meta = spr.csr_matrix(np.array(meta.values))
        return csr_meta

    def mf_(self, train_songs_A, train_tags_A, val_songs_A, val_tags_A, test_songs_A, test_tags_A, song_ntop = 500, tag_ntop = 50, iteration=20, score = False):
        
        print(f'MF... iters:{iteration}')

        val_song_res = []
        val_tag_res = []
        test_song_res = []
        test_tag_res = []

        songs_A = spr.vstack([val_songs_A,test_songs_A,train_songs_A])
        tags_A = spr.vstack([val_tags_A,test_tags_A,train_tags_A])

        als_model = ALS(factors=256, regularization=0.08, use_gpu=True, iterations=iteration) # epoch
        als_model.fit(songs_A.T * 100)
        als_model_tag = ALS(factors=32, regularization=0.08, use_gpu=True, iterations=iteration)
        als_model_tag.fit(tags_A.T * 100)

        def res_recommend(id, als_model = als_model, matrix = val_songs_A, N = song_ntop, nid_id = self.song_sid_id, id_index = self.plylst_val_song.index, res = val_song_res):
            try:
                cand_song = als_model.recommend(id, val_songs_A, N=N, filter_already_liked_items=True)

                rec_song_idx = [nid_id.get(x[0]) for x in cand_song]
                rec_song_score = [x[1] for x in cand_song]

                val_song_res.append({"id":self.plylst_nid_id[id_index[id]],\
                    "songs" : rec_song_idx, "songs_score": rec_song_score})
            
            except IndexError:
                pass
                
        if score is True:
            for id in tqdm(range(self.n_val_song)):
                res_recommend(id, als_model = als_model, matrix = val_songs_A, N=song_ntop, nid_id = self.song_sid_id,\
                     id_index = self.plylst_val_song.index, res = val_song_res)
                res_recommend(id, als_model = als_model_tag, matrix = val_tags_A, N=tag_ntop, nid_id = self.tag_tid_id,\
                     id_index = self.plylst_val_tag.index, res = val_tag_res)
                res_recommend(id, als_model = als_model, matrix = test_songs_A, N=song_ntop, nid_id = self.song_sid_id,\
                     id_index = self.plylst_test_song.index, res = test_song_res)
                res_recommend(id, als_model = als_model_tag, matrix = test_tags_A, N=tag_ntop, nid_id = self.tag_tid_id,\
                     id_index = self.plylst_test_tag.index, res = test_tag_res)

    
        else: # Score > False
            
            val_cand_song = als_model.recommend_all(val_songs_A, N=song_ntop, filter_already_liked_items=True)
            val_cand_tag = als_model_tag.recommend_all(val_tags_A, N=tag_ntop, filter_already_liked_items=True)
            test_cand_song = als_model.recommend_all(test_songs_A, N=song_ntop, filter_already_liked_items=True)
            test_cand_tag = als_model_tag.recommend_all(test_tags_A, N=tag_ntop, filter_already_liked_items=True)

            val_song_res = [{"id":self.plylst_nid_id[self.plylst_val_song.index[id]], "songs": [self.song_sid_id.get(x) for x in rec_idx.tolist()]} for id, rec_idx in enumerate(val_cand_song,0)]
            val_tag_res = [{"id":self.plylst_nid_id[self.plylst_val_tag.index[id]], "tags": [self.tag_tid_id.get(x) for x in rec_idx.tolist()]} for id,rec_idx in enumerate(val_cand_tag,0)]
            test_song_res = [{"id":self.plylst_nid_id[self.plylst_test_song.index[id]], "songs": [self.song_sid_id.get(x) for x in rec_idx.tolist()]} for id, rec_idx in enumerate(test_cand_song,0)]
            test_tag_res = [{"id":self.plylst_nid_id[self.plylst_test_tag.index[id]], "tags": [self.tag_tid_id.get(x) for x in rec_idx.tolist()]} for id,rec_idx in enumerate(test_cand_tag,0)]
          
        print("DONE")

        return val_song_res, val_tag_res, test_song_res, test_tag_res

    def multi_mf_(self, train_songs_A, train_tags_A, val_songs_A, val_tags_A, test_songs_A, test_tags_A, meta = True, song_ntop = 500, tag_ntop = 50, iteration=20, score = False):
        
        print(f'Multi_MF... iters:{iteration}')

        val_res = []
        test_res = []

        songs_A = spr.vstack([val_songs_A,test_songs_A,train_songs_A])
        tags_A = spr.vstack([val_tags_A,test_tags_A,train_tags_A])

        print(val_songs_A.shape, test_songs_A.shape, train_songs_A.shape)
        
        if meta == True:
            s_meta = self.mkspr_for_meta()
            print(songs_A.shape, tags_A.shape ,s_meta.shape)
            A = spr.hstack([songs_A, tags_A, s_meta])
        else:
            A = spr.hstack([songs_A, tags_A])

        als_model = ALS(factors=256, regularization=0.08, use_gpu=True, iterations=iteration)
        als_model.fit(A.T * 100)

        song_model = ALS(use_gpu=True)
        tag_model = ALS(use_gpu=True)

        song_model.user_factors = als_model.user_factors
        tag_model.user_factors = als_model.user_factors

        song_model.item_factors = als_model.item_factors[:self.n_songs]
        tag_model.item_factors = als_model.item_factors[self.n_songs:]

        # for val
        val_song_rec_csr = songs_A[:self.n_val, :]
        val_tag_rec_csr = tags_A[:self.n_val, :]

        # for test
        test_song_rec_csr = songs_A[self.n_val:self.n_val+self.n_test, :]
        test_tag_rec_csr = tags_A[self.n_val:self.n_val+self.n_test, :]

        if score is True:
            pass

        else:
            # val
            cand_song = song_model.recommend_all(val_song_rec_csr, N=song_ntop)
            cand_tag = tag_model.recommend_all(val_tag_rec_csr, N=tag_ntop)

            val_res = [{"id":self.plylst_nid_id[self.n_train +id],"songs": [self.song_sid_id.get(x) for x in rec_idx[0].tolist()], "tags":[self.tag_tid_id.get(x) for x in rec_idx[1].tolist()]} for id, rec_idx in enumerate(zip(cand_song,cand_tag))]

            # test
            cand_song = song_model.recommend_all(test_song_rec_csr, N=song_ntop)
            cand_tag = tag_model.recommend_all(test_tag_rec_csr, N=tag_ntop)

            test_res = [{"id": self.plylst_nid_id[self.n_train + self.n_val + id], "songs": [self.song_sid_id.get(x) for x in rec_idx[0].tolist()], "tags": [self.tag_tid_id.get(x) for x in rec_idx[1].tolist()]} for id, rec_idx in enumerate(zip(cand_song,cand_tag))]
        
        return val_res, test_res

    def make_result(self, song, tag):
        song_df = pd.DataFrame(song)
        tag_df = pd.DataFrame(tag)

        results = pd.merge(song_df, tag_df, on='id', how='outer')

        return results


                # val_song) 18636
                # val_tag) 11605
                # test_song) 8697
                # test_tag) 5422