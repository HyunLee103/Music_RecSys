import json
from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
import numpy as np
import pandas as pd
from tqdm import tqdm
import gensim  # 3.3.0 not available keyedvectors
from arena_util import load_json
from sklearn.cluster import KMeans
import pickle

"""
PlyEmbedding 이라는 클래스를 만들에정
-- attributes
self.data
self.s2v = None

-- methods
def make_s2v():
    pass

def song2vec():
    return self.s2v
    
def by_mean_s2v():
    if self.s2v:
        pass
        
def by_sum_s2v():
    if self.s2v:
        pass
        
def by_doc2vec():
    pass

def by_autoencoding():
    pass
"""



train_path = "arena_data/orig/train.json"

train = load_json(train_path)
len(train)

"""
W2V으로 song embedding.
convert songID to string.
remove 0 or 1 length. (1 has no context)

당연히 train에 없는 곡이거나 min_count에 안걸리는
노래들은 embedding 안된다.
"""
def get_s2v(data, min_count=5, size=100, window=5, sg=0):
    print("Playlist len:", len(data))
    songs = [list(map(str, x['songs'])) for x in data if len(x['songs']) > 1]
    print("Remove 0 or 1 song list:", len(songs))
    w2v = Word2Vec(songs, size=size, window=window,
                   min_count=min_count, sg=sg)
    return w2v


s2v = get_s2v(train)

"""
pairs of
<ID : playlist vector>

Enable to find plyIDs similar with new playlist vectors
"""
def get_p2v(playlists, w2v, mode='mean'):
    ply_id = []
    ply_vec = []
    for p in tqdm(playlists):
        tmp_list = []
        for song in p['songs']:
            try:
                tmp_list.append(w2v.wv.get_vector(str(song)))
            except KeyError:
                pass
        if tmp_list != []:
            ply_id.append(str(p['id']))
            ply_vec.append(np.mean(tmp_list, axis=0))
    out = WordEmbeddingsKeyedVectors(vector_size=100)
    out.add(ply_id, ply_vec)
    
    print("p2v len:", len(ply_id))
    
    return out

p2v = get_p2v(train, s2v)
p2v.similar_by_vector(p2v['147668'], topn=3)

"""
for check p2v performance
"""
def get_p2v_title(playlists, w2v, mode='mean'):
    ply_t = []
    ply_vec = []
    for p in tqdm(playlists):
        tmp_list = []
        for song in p['songs']:
            try:
                tmp_list.append(w2v.wv.get_vector(str(song)))
            except KeyError:
                pass
        if tmp_list != []:
            ply_t.append(p['plylst_title'])
            ply_vec.append(np.mean(tmp_list, axis=0))
    out = WordEmbeddingsKeyedVectors(vector_size=100)
    out.add(ply_t, ply_vec)
    
    print("p2v len:", len(ply_t))
    
    return out

p2v_title = get_p2v_title(train, s2v)
p2v_title.most_similar('밤에 듣기 좋은 음악들', topn=5)
# shit no future, no hope

"""
for training KMeans.
return pairs of <Title : plylst vector>
"""
def get_title_w_vec(playlists, w2v):
    titles = []
    vectors = []
    for p in tqdm(playlists):
        tmp_list = []
        for song in p['songs']:
            try:
                tmp_list.append(w2v.wv.get_vector(str(song)))
            except KeyError:
                pass
        if tmp_list != []:
            titles.append(p['plylst_title'])
            vectors.append(np.mean(tmp_list, axis=0))
    
    print("length:", len(titles))
    
    return titles, vectors

titles, vectors = get_title_w_vec(train, s2v)


"""
Clustering playlist vectors
Extra ordinary slow...
"""
n_clusters = 500
cluster_model = KMeans(n_clusters=n_clusters,
                       verbose=1, max_iter=50)

# very slow-----
out = cluster_model.fit_predict(vectors)

save = dict(zip(titles, out))

# load clustering result    
with open("cluster_result", "rb") as f:
    tmp = pickle.load(f)
    
"""
FastText
"""

tmp = pd.DataFrame({'title':titles,
                    'cluster':out})

result = tmp.groupby('cluster')['title'].apply(list)
sentences = result.tolist()

ft_model = FastText(sentences)

ft_model.wv.most_similar("가을밤", topn=10)
# It needs pre-processing