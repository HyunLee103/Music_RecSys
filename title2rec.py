from ply_tag_embedding import PlyEmbedding
import numpy as np
import pandas as pd
from gensim.models import FastText
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
from tqdm import tqdm
import gensim  # 3.3.0 not available keyedvectors
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import re
from arena_util import load_json, write_json, most_popular, remove_seen
import math
# from khaiii import KhaiiiApi

class Title2Rec:
    def __init__(self):
        super().__init__()
        self.cluster_model = None
        self.fasttext = None
        self.t2r = None
        self.good_tags = ['NNG', 'NNP', 'NNB', 'NP', 'NR',
        'VA', 'MAG', 'SN', 'SL']
        self.khaiii = KhaiiiApi()
    
    ## fit clustering
    def fit_clustering(self, vectors,
                   n_clusters, verbose=0, max_iter=50):
        self.cluster_model = KMeans(n_clusters=n_clusters, verbose=verbose,
                            max_iter=max_iter)
        print("Data length: ", len(vectors))
        print("Fit KMeans...")
        self.cluster_model.fit(vectors)
        print("done.")

    ## preprocess for clustering
    def preprocess_clustering(self, titles, vectors, ID=True, khaiii=True, verbose=False):
        ## t: title / v: vectors / i : plylst id
        if ID:
            id_list = list(map(lambda x: x.split(' ')[0][1:-1], titles))
            titles = list(map(lambda x: ' '.join(x.split(' ')[1:]), titles))
        else:
            id_list = list(range(len(titles)))
        t_v = list(zip(titles, vectors, id_list))
        stable = [(t, v, i) for t, v, i in t_v if re.findall('[가-힣a-zA-Z&]+', t) != []]
        stable = [(' '.join(re.findall('[가-힣a-zA-Z&]+|90|80|70', t)), v, i) for t, v, i in stable]
        stable = [(t, v, i) for t, v, i in stable if t != '']

        ## title morph analysis by Khaiii
        def tag_process(title, khaiii, good_tags):
            token = khaiii.analyze(title)
            ## join : space bar between list element
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

    ## cleansing text before Khaiii
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

    ## predict cluster with cluster model, return clusters sorted by distance
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

    ## mk Fasttext model with cluster(500)
    def fit_fasttext(self, data):
        sentence = data.groupby('cluster')['title'].apply(list).tolist()
        print("Fit fasttext...")
        self.fasttext = FastText(sentence)
        print('done.')

    ## mk title2rec model
    def fit_title2rec(self, titles, ID):
        keys = [i + " " + t for t, i in zip(titles, ID)]
        print('Fit title2rec...')
        vectors = list(map(self.fasttext.wv.get_vector, titles))
        self.t2r = WordEmbeddingsKeyedVectors(vector_size=100)
        self.t2r.add(keys, vectors)
        print('done.')

    ## get title vectors from fasttext model ( most similar 10 - default)
    def forward(self, titles, topn=10):
        ft = list(map(self.fasttext.wv.get_vector, titles))
        out = [self.t2r.wv.similar_by_vector(t, topn=topn) for t in ft]
        return out

    ## load cluster model
    def load_cluster(self, fname):
        self.cluster_model = joblib.load(fname)
        print("load complete")

    ## load fasttext model
    def load_fasttext(self, path):
        self.fasttext = gensim.models.FastText.load(path)

    ## load title to songs model
    def load_t2r(self, path):
        self.t2r = gensim.models.KeyedVectors.load(path)

    def title2rec(self, ply, song_n, tag_n, song_const, tag_const, khaiii=True):
        title, _, _ = self.preprocess_clustering([ply['plylst_title']], [None], ID=False, khaiii=khaiii, verbose=False)
        if title == []:
            if ply['tags'] != []:
                return ply['songs'], ply['tags'], 1, 0
            else:
                return ply['songs'], ply['tags'], 1, 1

        title = title[0]
        similars = self.forward([title], topn=200)[0]

        ID = [int(sim[0].split(" ")[0]) for sim in similars]
        similar = [sim[1] for sim in similars]

        tmp_df = pd.DataFrame({'id':ID, 'similar':similar})
        tmp_df = pd.merge(tmp_df, train_df[['id', 'songs', 'tags']], how='left', on='id')
        tmp_df['song_len'] = tmp_df['songs'].apply(len)
        tmp_df['song_len'] = tmp_df['song_len'].cumsum().shift(1).fillna(0)
        song_df = tmp_df[tmp_df['song_len'] < 2000]

        score_dict = {}
        for sim, songs in zip(song_df['similar'], song_df['songs']):
            for i, song in enumerate(songs):
                score = (-math.log(i+1, 2) + song_const) * sim
                try:
                    score_dict[song] += score
                except KeyError:
                    score_dict[song] = score

        pick = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        pick = [p[0] for p in pick]
        song_res = pick[:song_n]
        # date = pd.to_datetime(ply['updt_date'])
        # pick = [p for p in pick if song_date[p] <= date]
        # song_res = pick[:song_n]

        if len(song_res) < song_n:
            song_df = tmp_df[tmp_df['song_len'] >= 2000]
            for sim, songs in zip(song_df['similar'], song_df['songs']):
                for i, song in enumerate(songs):
                    score = (-math.log(i+1, 2) + song_const) * sim
                    try:
                        score_dict[song] += score
                    except KeyError:
                        score_dict[song] = score
            pick = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
            pick = [p[0] for p in pick]
            # pick = [p for p in pick if song_date[p] <= date]
            song_res = pick[:song_n]
        
        # assert len(song_res) == song_n

        # song_res = [p[0] for p in pick]
        
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

## mk cluster model 
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

    embed = PlyEmbedding()
    embed.load_s2v(s2v_path)
    titles, vectors = embed.song_based(mode='s2v', by='mean', keyedvector=False) ## title, vectors not in KeydVectos
    # T2R
    t2r = Title2Rec()
    # remove non alpha or hangul. tokenize, ( t : title, v : vectors, ID : plylst id )
    t, v, ID = t2r.preprocess_clustering(titles, vectors, ID=True, khaiii=True, verbose=True)
   
    t2r.fit_clustering(v, n_clusters=500)
    joblib.dump(t2r.cluster_model, "cluster_model/tvt_500c_s2v_khaiii.pkl")