import json
from os import write
from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
import numpy as np
import pandas as pd
from tqdm import tqdm
import gensim  # 3.3.0 not available keyedvectors
from arena_util import load_json, write_json, most_popular, remove_seen
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import sklearn.metrics
import pickle   
import re
from collections import Counter
from khaiii import KhaiiiApi
import math


"""
class PlyEmbedding -> 플레이리스트 들어가서 플레이리스트 마다 일정하게 벡터가 부여된다.

class Title2Rec -> 벡터가 부여된 타이틀이 들어가서 cluster, fasttext, T2R
"""

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

    def load_s2v(self, path):
        self.s2v = gensim.models.KeyedVectors.load(path)
        print("load complete.")

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


# data load
train_path = "arena_data/orig/train.json"

train = load_json(train_path)

embed = PlyEmbedding(train)

# make Word2vec & Doc2vec
embed.make_s2v()
embed.make_d2v()

# load s2v models
embed.load_s2v("embedding_models/s2v.model")


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
        self.good_tags = ['NNG', 'NNP', 'NNB', 'NP', 'NR',
        'VA', 'MAG', 'SN', 'SL']
        self.khaiii = KhaiiiApi()
        
    def fit_clustering(self, vectors,
                   n_clusters, verbose=0, max_iter=50):
        self.cluster_model = KMeans(n_clusters=n_clusters, verbose=verbose,
                            max_iter=max_iter)
        print("Data length: ", len(vectors))
        print("Fit KMeans...")
        self.cluster_model.fit(vectors)
        print("done.")
        
    def preprocess_clustering(self, titles, vectors, ID=True, khaiii=True, verbose=False):
        if ID:
            id_list = list(map(lambda x: x.split(' ')[0][1:-1], titles))
            titles = list(map(lambda x: ' '.join(x.split(' ')[1:]), titles))
        else:
            id_list = list(range(len(titles)))
        t_v = list(zip(titles, vectors, id_list))
        stable = [(t, v, i) for t, v, i in t_v if re.findall('[가-힣a-zA-Z&]+', t) != []]
        stable = [(' '.join(re.findall('[가-힣a-zA-Z&]+|90|80|70', t)), v, i) for t, v, i in stable]
        stable = [(t, v, i) for t, v, i in stable if t != '']

        def tag_process(title, khaiii, good_tags):
            token = khaiii.analyze(title)
            return ' '.join([morph.lex for to in token for morph in to.morphs if morph.tag in good_tags])

        if khaiii:
            if verbose:
                stable = [(tag_process(t, self.khaiii, self.good_tags), v, i) for t, v, i in tqdm(stable)]
                stable = [(t, v, i) for t, v, i in stable if t != '']
            else:
                stable = [(tag_process(t, self.khaiii, self.good_tags), v, i) for t, v, i in stable]
                stable = [(t, v, i) for t, v, i in stable if t != '']
        titles = [t for t, v, i in stable]
        vectors = [v for t, v, i in stable]
        id_list = [i for t, v, i in stable]
        if verbose:
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

    def load_cluster(self, fname):
        self.cluster_model = joblib.load(fname)
        print("load complete")

    def load_fasttext(self, path):
        self.fasttext = gensim.models.FastText.load(path)

    def load_t2r(self, path):
        self.t2r = gensim.models.KeyedVectors.load(path)


a = gensim.models.FastText.load("embedding_models/sub_train_fasttext.model")
b = gensim.models.KeyedVectors.load()


t2r = Title2Rec()

"""
1. 제목에 영어와 완전한 한글만 있는거(ex.2019.02.23, 123231-1, ㅋㅋ, ㅎㅎ, ^^, (하트하트))
2. 숫자와 특수문자 제거 (ex. 크리스마스 223) - 70, 80, 90 만 넣기
"""

t, v, ID = t2r.preprocess_clustering(titles, vectors, ID=True, khaiii=True)

#t2r.fit_clustering(v, n_clusters=500)

#joblib.dump(t2r.cluster_model, "cluster_model/sub_train_500c_s2v_khaiii.pkl")


t2r.load_cluster('cluster_model/sub_train_500center_khaiii.pkl')

data = t2r.pre_fasttext(t, v)

t2r.fit_fasttext(data)
t2r.fit_title2rec(t, ID)

t2r.fasttext.save("embedding_models/sub_train_fasttext.model")
t2r.t2r.save("embedding_models/sub_train_t2r.model")

t[32165]
similar = t2r.forward(['년 월'])
similar[0]

titles[20024]
khiii = KhaiiiApi()
[y.tag for x in khiii.analyze("슬플 때") for y in x.morphs]

"""
input: titles, n_song

T2R로 채워 넣을 곡 개수: ? (아마 30~40개) -> 이것도 input.

1. 현상황. title이 들어가면 근처 플레이리스트 n개를 train셋에서 찾아준다.
 - 라디오 63개.

1이면 다 넣어.
 
 a. n 개의 플레이리스트를 뽑는경우
    - n을 초과하게 거리가 1인 플레이리스트가 있는 경우.
       - like cnt로 sort해서 n개. 
       
 b. 거리의 threadhold. th <th> 이상
    - 한개도 안나올 수가 있다. -> 제일가까운거 하나만. input곡이 안되면 다음 것까지
    - 엄청 많이 나올 수도 있다.'
    
 c. input의 10배 곡 개수가 나올 때까지 플레이리스트 넘기기.


2. 플레이리스트는 찾았다. 곡은 어떻게 sorting 할 것인가?
   확정. 현 교수 공식 -> 유사도 곱하기 -> 더하기 -> sort
 
3. 형태소분석 (형, 명, 부) 
 
4. 평가방법
 - 제목만 있는 플레이리스트에 most_popular vs T2R
 

&*@#&*$ <<<태그 가져오는 것도 잊지 말자>>> ^&#*^


<7/8>
매번 s2v으로 한 플레이리스트 임베딩이 달라짐
달라진 것으로 clustering하면 결과 역시 달라지기 때문에
cluster 모델만 저장하는건 상당히 이상한 일이죠!
fit_fasttext에서 cluster를 예측할 때 매번 달라집니다.

a. 매번 cluster한다

b. 그에 따른 title의 벡터 값을 저장한다.

<회의>
1. Song2Vec을 저장해놓다. sub-train, train
2. khaiii vs re cluster 결과를 저장해놓자.
3. fasttext 오래 안걸리
4. 

"""
#######################################################################

train_path = "res/train.json"
val_path = "res/val.json"
s2v_path = "embedding_models/full_s2v.model"
cluster_path = "cluster_model/full_500c_s2v_khaiii.pkl"

# load data
train = load_json(train_path)
val = load_json(val_path)

# train to df
train_df = pd.DataFrame(train)

# embedding load
embed = PlyEmbedding(train)
embed.load_s2v(s2v_path)
#embed.make_s2v()
#embed.s2v.save("embedding_models/full_s2v.model")

# p2v for tag_by_song, title-vector for t2r
p2v = embed.song_based(mode='s2v', by='mean', keyedvector=True)
titles, vectors = embed.song_based(mode='s2v', by='mean', keyedvector=False)

# T2R
t2r = Title2Rec()

# remove non alpha or hangul. tokenize
t, v, ID = t2r.preprocess_clustering(titles, vectors, ID=True, khaiii=True, verbose=True)

# t2r.fit_clustering(v, n_clusters=500)
# joblib.dump(t2r.cluster_model, "cluster_model/full_500c_s2v_khaiii.pkl")

# load cluster
t2r.load_cluster(cluster_path)

# sort by cluster & distance from center
data = t2r.pre_fasttext(t, v)

# fit fasttext & title2rec
t2r.fit_fasttext(data)
t2r.fit_title2rec(t, ID)

# most popular
_, pop_songs = most_popular(train, 'songs', 100)
_, pop_tags = most_popular(train, 'tags', 10)

def put_most_popular(seq, pop):
    unseen = remove_seen(seq, pop)
    return seq + unseen[:len(pop) - len(seq)]

song_const = 7.66
tag_const = 3.9

def tag_by_songs(ply, n, const):
    songs = ply['songs']
    tag_const = const
    vec = []
    for song in songs:
        try:
            tmp = embed.s2v.wv.get_vector(str(song))
            vec.append(tmp)
        except KeyError:
            pass
    if vec == []:
        return []
    vec = np.mean(vec, axis=0)

    similars = p2v.wv.similar_by_vector(vec, topn=150)

    ID = [int(sim[0].split(" ")[0][1:-1]) for sim in similars]
    similar = [sim[1] for sim in similars]

    tmp_df = pd.DataFrame({'id':ID, 'similar':similar})
    tmp_df = pd.merge(tmp_df, train_df[['id', 'tags']], how='left', on='id')
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

    return res

def title2rec(ply, song_n, tag_n, song_const, tag_const, khaiii=True):
    title, _, _ = t2r.preprocess_clustering([ply['plylst_title']], [None], ID=False, khaiii=khaiii, verbose=False)
    if title == []:
        if ply['tags'] != []:
            return ply['songs'], ply['tags'], 1, 0
        else:
            return ply['songs'], ply['tags'], 1, 1

    title = title[0]
    similars = t2r.forward([title], topn=150)[0]

    ID = [int(sim[0].split(" ")[0]) for sim in similars]
    similar = [sim[1] for sim in similars]

    tmp_df = pd.DataFrame({'id':ID, 'similar':similar})
    tmp_df = pd.merge(tmp_df, train_df[['id', 'songs', 'tags']], how='left', on='id')
    tmp_df['song_len'] = tmp_df['songs'].apply(len)
    tmp_df['song_len'] = tmp_df['song_len'].cumsum().shift(1).fillna(0)
    song_df = tmp_df[tmp_df['song_len'] < 1500]

    score_dict = {}
    for sim, songs in zip(song_df['similar'], song_df['songs']):
        for i, song in enumerate(songs):
            score = (-math.log(i+1, 2) + song_const) * sim
            try:
                score_dict[song] += score
            except KeyError:
                score_dict[song] = score

    pick = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:song_n]
    song_res = [p[0] for p in pick]

    if ply['tags'] != []:
        return song_res, ply['tags'], 1, 0

    tmp_df['tag_len'] = tmp_df['tags'].apply(len)
    tmp_df['tag_len'] = tmp_df['tag_len'].cumsum().shift(1).fillna(0)
    tag_df = tmp_df[tmp_df['tag_len'] < 150]

    score_dict = {}
    for sim, tags in zip(tag_df['similar'], tag_df['tags']):
        for i, tag in enumerate(tags):
            score = (-math.log(i+1, 2) + tag_const) * sim
            try:
                score_dict[tag] += score
            except KeyError:
                score_dict[tag] = score

    pick = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:tag_n]
    tag_res = [p[0] for p in pick]

    return song_res, tag_res, 1, 1


for ply in tqdm(val):
    ply['song_dirty'] = 0
    ply['tag_dirty'] = 0

    if ply['songs'] != []:
        if ply['tags'] != []:
            pass
        else:
            ply['tags'] = tag_by_songs(ply, 10, 3.9)
            if len(ply['tags']) < 10:
                ply['tags'] = put_most_popular(ply['tags'], pop_tags)
            ply['tag_dirty'] = 1

    else:
        songs, tags, song_sign, tag_sign = title2rec(ply, 100, 10, song_const, tag_const)
        if (song_sign) and (len(songs) < 100):
            songs = put_most_popular(songs, pop_songs)
        if (tag_sign) and (len(tags) < 10):
            tags = put_most_popular(tags, pop_tags)
        ply['songs'] = songs
        ply['tags'] = tags
        ply['song_dirty'] = song_sign
        ply['tag_dirty'] = tag_sign

write_json(val, "val_t2r_all.json")

for x in val:
    if (x['song_dirty']) and (len(x['songs']) != 100):
        print('fuck')
    if (x['tag_dirty']) and (len(x['tags']) != 10):
        print('fuck')


result = load_json("/mnt/c/users/koo/Desktop/results.json")

for i, ply in enumerate(val):                                                           
    if ply['song_dirty']:
        result[i]['songs'] = ply['songs']
    if ply['tag_dirty']:
        result[i]['tags'] = ply['tags']

write_json(result, "1500_150/results.json")

write_json(val, "val_1500_150_mostpop/val.json")