import pandas as pd
import numpy as np
from ply_tag_embedding import PlyEmbedding
from title2rec import Title2Rec
from arena_util import load_json, write_json, most_popular, remove_seen
import math
from tqdm import tqdm

"""
case 1 : O O  ( mf or autoencoder)
case 2 : O X  ( X : predict tag by song, O : mf )
case 3 : X O  ( X : predict song by tag, O : mf )
case 4 : X X  ( X : title2song, title2tag)
"""

def put_most_popular(seq, pop):
    unseen = remove_seen(seq, pop)
    return seq + unseen[:len(pop) - len(seq)]

## fill_X for case2, case3, case4
def fill_X(train, val):
    # embedding load ( need to mk s2v model by ply_tag_embedding.py )
    embed = PlyEmbedding(train)
    embed.load_s2v(s2v_path)
    # p2v for tag_by_song, title-vector for t2r
    titles, vectors = embed.song_based(mode='s2v', by='mean', keyedvector=False) ## title, vectors not in KeydVectos

    # T2R
    t2r = Title2Rec()
    # remove non alpha or hangul. tokenize, ( t : title, v : vectors, ID : plylst id )
    t, v, ID = t2r.preprocess_clustering(titles, vectors, ID=True, khaiii=True, verbose=True)

    # load cluster ( need to mk cluster pkl file )
    t2r.load_cluster(cluster_path)

    # sort by cluster & distance from center
    data = t2r.pre_fasttext(t, v)

    # fit fasttext & title2rec
    t2r.fit_fasttext(data)
    t2r.fit_title2rec(t, ID)

    # most popular
    _, pop_songs = most_popular(train, 'songs', 100)
    _, pop_tags = most_popular(train, 'tags', 10)

    for ply in tqdm(val):
        ply['song_dirty'] = 0
        ply['tag_dirty'] = 0

        if ply['songs'] != []:
            if ply['tags'] != []:
                pass
            else:
                ply['tags'] = embed.tag_by_songs(ply, 10, 3.9)
                if len(ply['tags']) < 10:
                    ply['tags'] = put_most_popular(ply['tags'], pop_tags)
                ply['tag_dirty'] = 1

        else:
            songs, tags, song_sign, tag_sign = t2r.title2rec(ply, 100, 10, song_const, tag_const)
            if (song_sign) and (len(songs) == 0):
                songs = put_most_popular(songs, pop_songs)
                #raise RuntimeError("song length < 100")
            if (tag_sign) and (len(tags) < 10):
                tags = put_most_popular(tags, pop_tags)
            ply['songs'] = songs
            ply['tags'] = tags
            ply['song_dirty'] = song_sign
            ply['tag_dirty'] = tag_sign
    
    return val




if __name__ == '__main__':

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

    # custom grade
    song_const = 7.66
    tag_const = 3.9

    ## fill_X ( case2, case3, case4 )
    val = fill_X(train, val)




