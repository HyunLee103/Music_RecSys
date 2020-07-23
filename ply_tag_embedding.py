import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
from tqdm import tqdm
import gensim  # 3.3.0 not available keyedvectors
import math
from arena_util import load_json, write_json, most_popular, remove_seen

class PlyEmbedding:
    def __init__(self, data):
        super().__init__()
        self.data = data  # for song based embedding
        self.data_df = pd.DataFrame(self.data)
        self.s2v = None
        print("Data length:", len(data))
    
    ## s2v ( make song embedding model )
    def make_s2v(self, min_count=5, size=100, window=5, sg=0):
        songs = [list(map(str, x['songs'])) for x in self.data if len(x['songs']) > 1]
        print("Original data length: ", len(self.data))
        print("Playlist count after remove 0 or 1: ", len(songs))
        print("Traning song2vec...")
        self.s2v = Word2Vec(songs, size=size, window=window,
                            min_count=min_count, sg=sg)
        print("done.")
    
    ## call s2v model
    def get_song2vec(self):
        return self.s2v
    
    ## plylst vector ( all )
    def song_based(self, mode='s2v', by='mean', keyedvector=True):
        if not self.s2v:
            print("Song2Vec not exist.\nRun make_s2v first.")
            return
            
        if not by in ['mean', 'sum']:
            raise RuntimeError("'by' gets 'mean' or 'sum'")
        
        ply_id = []
        ply_vec = []
        
        for p in tqdm(self.data):
            if by == 'mean':
                tmp = []
            else:
                tmp = 0
            for song in p['songs']:
                try:
                    if by == 'mean':
                        tmp.append(self.s2v.wv.get_vector(str(song)))
                    else:
                        tmp += self.s2v.wv.get_vector(str(song))
                except KeyError:
                    pass
            if by == 'mean':
                if tmp != []:
                    ply_id.append('(' + str(p['id']) + ') ' + p['plylst_title'])
                    ply_vec.append(np.mean(tmp, axis=0))
            else:
                if type(tmp) != int:
                    ply_id.append('(' + str(p['id']) + ') ' + p['plylst_title'])
                    ply_vec.append(tmp)
        
        print("Original data length: ", len(self.data))
        print("Embedded data length: ", len(ply_id))
        
        if not keyedvector:
            return ply_id, ply_vec
        
        out = WordEmbeddingsKeyedVectors(vector_size=100)
        out.add(ply_id, ply_vec)
        
        return out

    ## load song embedding model
    def load_s2v(self, path):
        self.s2v = gensim.models.KeyedVectors.load(path)
        print("load complete.")

    ## get plylst embedding vector from songs ( by mean )
    def get_embed(self, songs):
        out = []
        for song in songs:
            try:
                tmp = self.s2v.wv.get_vector(str(song))
                out.append(tmp)
            except KeyError:
                pass
        if out == []:
            return False
        else:
            return np.mean(out, axis=0)  

    ## predict song by tags ( case 3 in main.py )
    def tag_by_songs(self, ply, n , const = 3.9):
        songs = ply['songs']
        tag_const = const
        vec = self.get_embed(songs)
        similars = self.s2v.wv.similar_by_vector(vec, topn=150)

        ID = [int(sim[0].split(" ")[0][1:-1]) for sim in similars]  # similars : 150
        similar = [sim[1] for sim in similars]

        tmp_df = pd.DataFrame({'id':ID, 'similar':similar})
        tmp_df = pd.merge(tmp_df, self.data_df[['id', 'tags']], how='left', on='id')
        tmp_df['len'] = tmp_df['tags'].apply(len)
        tmp_df['len'] = tmp_df['len'].cumsum().shift(1).fillna(0)
        tmp_df = tmp_df[tmp_df['len'] < 150]

        score_dict = {}
        for sim, tags in zip(tmp_df['similar'], tmp_df['tags']):
            for i, tag in enumerate(tags):
                score = (-math.log(i+1, 2) + tag_const) * sim
                try:
                    score_dict[tag] += score
                except KeyError:
                    score_dict[tag] = score

        pick = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:n]
        res = [p[0] for p in pick]
        score_res = [p[1] for p in pick]

        return res, score_res

## mk and save song embedding(s2v) model
if __name__ == "__main__":

    # paths
    train_path = "res/train.json"  # original train file
    val_path = "res/val.json"   # original validation file
    test_path = "res/test.json" # original test file
    meta_path = "res/song_meta.json"    # song_meta.json
    s2v_path = "pretrained/tvt_s2v.model" # train, valid, test song embedding model
    cluster_path = "pretrained/tvt_500c_s2v_khaiii.pkl"  # train, valid, test 500 cluster model

    # load data
    train = load_json(train_path)
    val = load_json(val_path)
    test = load_json(test_path)
    song_meta = load_json(meta_path)

    # train to df
    train_df = pd.DataFrame(train)

    # embedding load
    embed = PlyEmbedding(train)
    # embed.load_s2v(s2v_path)
    embed.make_s2v()
    embed.s2v.save("embedding_models/tvt_s2v.model")

"""
plylst update date랑 song issue date에 시간적 모순이 발생하는 경우를 제외하고 추천 진행
# make song to date dictionary
meta_df = pd.DataFrame(song_meta)
song_date = meta_df[['id', 'issue_date']]
song_date['issue_date'] = song_date['issue_date'].apply(lambda x: '19500101' if x == '00000000' else x)
song_date['issue_date'] = song_date['issue_date'].apply(lambda x: x[:4] + '01' + x[-2:] if x[-4:-2] == '00' else x)
song_date['issue_date'] = song_date['issue_date'].apply(lambda x: x[:6] + '01' if x[-2:] == '00' else x)
song_date.iloc[168071, 1] = '20010930'
song_date.iloc[430200, 1] = '20060131'
song_date.iloc[692325, 1] = '20100513'
song_date['issue_date'] = pd.to_datetime(song_date['issue_date'])
song_date = {i: date for i, date in zip(song_date['id'], song_date['issue_date'])}
"""


"""
def __init__
    - data: for song based embedding

def make_s2v
    - params: same with Word2Vec params
    
def make_d2v
    - vector_size: vector size of the Doc2Vec model
    
def song_based
    - mode: 's2v'(Word2Vec base) or 'd2v'(Doc2Vec base)
    - by:   'mean' or 'sum'
    - keyedvector: True -> return Word2Vec type
                False-> return (id+titles, vectors) : list
                플레이리스트 벡터로 근처 플레이리스트 찾을 땐 True
                클러스터링 학습을 하고 싶을 땐 False
"""
