import json
from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
import numpy as np
import pandas as pd
from tqdm import tqdm
import gensim  # 3.3.0 not available keyedvectors
from arena_util import load_json
from sklearn.cluster import KMeans
import sklearn.metrics
import pickle
import re
from collections import Counter


class PlyEmbedding:
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
    
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.s2v = None
        self.d2v = None
        print("Data length:", len(data))
        
    def make_s2v(self, min_count=5, size=100, window=5, sg=0):
        songs = [list(map(str, x['songs'])) for x in self.data if len(x['songs']) > 1]
        print("Original data length: ", len(self.data))
        print("Playlist count after remove 0 or 1: ", len(songs))
        print("Traning song2vec...")
        self.s2v = Word2Vec(songs, size=size, window=window,
                            min_count=min_count, sg=sg)
        print("done.")
        
    def get_song2vec(self):
        return self.s2v
        
    def make_d2v(self, vector_size=100):
        doc = [TaggedDocument(list(map(str, x['songs'])),
                        ['(' + str(x['id']) + ') ' + x['plylst_title']]) for x in self.data]
        
        print("Training Doc2Vec...")
        self.d2v = Doc2Vec(doc, vector_size=vector_size, workers=4)
        print('done')
        
    def get_doc2vec(self):
        return self.d2v
    
    def song_based(self, mode='s2v', by='mean', keyedvector=True):
        if mode == 's2v':
            if not self.s2v:
                print("Song2Vec not exist.\nRun make_s2v first.")
                return
        elif mode == 'd2v':
            if not self.d2v:
                print("Doc2Vec not exist.\nRun make_d2v first.")
                return
        else:
            print("mode gets 's2v' or 'd2v'")
            
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
                        if mode == 's2v':
                            tmp.append(self.s2v.wv.get_vector(str(song)))
                        else:
                            tmp.append(self.d2v.wv.get_vector(str(song)))
                    else:
                        if mode == 's2v':
                            tmp += self.s2v.wv.get_vector(str(song))
                        else:
                            tmp += self.d2v.wv.get_vector(str(song))
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
        
    def by_autoencoder(self):
        pass


# data load
train_path = "arena_data/orig/train.json"

train = load_json(train_path)

embed = PlyEmbedding(train)

# make song2vec & doc2vec
embed.make_s2v()
embed.make_d2v()

# all possible plylst embedding
s2v_sum = embed.song_based(mode='s2v', by='sum', keyedvector=True)
s2v_mean = embed.song_based(mode='s2v', by='mean', keyedvector=True)
d2v_sum = embed.song_based(mode='d2v', by='sum', keyedvector=True)
d2v_mean = embed.song_based(mode='d2v', by='mean', keyedvector=True)

# check the result.
def p2v_test(data, num, m1, m2):
    print(data[num]['id'], ' '+data[num]['plylst_title'])
    res1 = m1.most_similar(f"({data[num]['id']}) {data[num]['plylst_title']}")
    res2 = m2.most_similar(f"({data[num]['id']}) {data[num]['plylst_title']}")
    print("model 1")
    print("-"*20)
    for i in res1:
        print(i[0])
    print()
    print("model 2")
    print("-"*20)
    for j in res2:
        print(j[0])

p2v_test(train, 8, s2v_sum, s2v_mean)

"""
W2V으로 song embedding.
convert songID to string.
remove 0 or 1 length. (1 has no context)

당연히 train에 없는 곡이거나 min_count에 안걸리는
노래들은 embedding 안된다.
"""

"""
<6/27>
클러스터링 성능 향상을 위해 미리
숫자와 특수문자로 구성된 걸 빼자!

우리: [['가을감성 노래~', '가을감성 발라드!', '겨울 감성 발라드'], ]
보통의 w2v: [['나는', '밥을', '먹었다'], ]

클러스터링 할 때 센터에서 거리 순으로 문장 배열!
"""


titles, vectors = embed.song_based(keyedvector=False)


class Title2Rec:
    def __init__(self):
        super().__init__()
        self.cluster_model = None
        self.fasttext = None
        self.t2r = None
        
    def fit_clustering(self, vectors,
                   n_clusters, verbose=0, max_iter=50):
        self.cluster_model = KMeans(n_clusters=n_clusters, verbose=verbose,
                            max_iter=max_iter)
        print("Data length: ", len(vectors))
        print("Fit KMeans...")
        self.cluster_model.fit(vectors)
        print("done.")
        
    @staticmethod
    def preprocess_clustering(titles, vectors, ID=True):
        if ID:
            id_list = list(map(lambda x: x.split(' ')[0][1:-1], titles))
            titles = list(map(lambda x: ' '.join(x.split(' ')[1:]), titles))
        t_v = list(zip(titles, vectors, id_list))
        stable = [(t, v, i) for t, v, i in t_v if re.findall('[가-힣a-zA-Z&]+', t) != []]
        stable = [(' '.join(re.findall('[가-힣a-zA-Z&]+|90|80|70', t)), v, i) for t, v, i in stable]
        stable = [(t, v, i) for t, v, i in stable if t != '']
        titles = [t for t, v, i in stable]
        vectors = [v for t, v, i in stable]
        id_list = [i for t, v, i in stable]
        print("Original lenght: ", len(t_v))
        print("Processed length: ", len(titles))
        
        return titles, vectors, id_list
   
    @staticmethod
    def text_process(titles, ID=True):
        if ID:
            titles = list(map(lambda x: ' '.join(x.split(' ')[1:]), titles))
        stable = [x for x in titles if re.findall('[가-힣a-zA-Z&]+', x) != []]
        stable = [' '.join(re.findall('[가-힣a-zA-Z&]+|90|80|70', x)) for x in stable]
        stable = [x for x in stable if x != '']
        print("Only hangul & alpha & and sign.")
        print("Original lenght: ", len(titles))
        print("Processed length: ", len(stable))
        
        return stable

    def pre_fasttext(self, titles, vectors):
        if not self.cluster_model:
            raise RuntimeError("Please fit clustering model.")
        cluster_out = self.cluster_model.predict(vectors)
        transform = self.cluster_model.transform(vectors)
        dist = [distance[cluster] for cluster, distance in zip(cluster_out, transform)]
        data = pd.DataFrame({'title': titles,
                             'cluster': cluster_out,
                             'distance': dist})
        return data.sort_values(['cluster', 'distance'])
    
    def fit_fasttext(self, data):
        sentence = data.groupby('cluster')['title'].apply(list).tolist()
        print("Fit fasttext...")
        self.fasttext = FastText(sentence)
        print('done.')
        
    def fit_title2rec(self, titles, ID):
        keys = [i + " " + t for t, i in zip(titles, ID)]
        print('Fit title2rec...')
        vectors = list(map(self.fasttext.wv.get_vector, titles))
        self.t2r = WordEmbeddingsKeyedVectors(vector_size=100)
        self.t2r.add(keys, vectors)
        print('done.')
    
    def forward(self, titles, topn=10):
        ft = list(map(self.fasttext.wv.get_vector, titles))
        out = [self.t2r.wv.similar_by_vector(t, topn=topn) for t in ft]
        return out

t2r = Title2Rec()

t, v, ID = Title2Rec.preprocess_clustering(titles, vectors, ID=True)


t2r.fit_clustering(v[:1000], n_clusters=200)

data = t2r.pre_fasttext(t[:1000], v[:1000])

t2r.fit_fasttext(data)
t2r.fit_title2rec(t, ID)

similar = t2r.forward(['기분좋은 봄날', '뜨거운 여름밤'])


