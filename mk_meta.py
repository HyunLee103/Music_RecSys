import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class mk_meta:
    def __init__(self):
        self.train = pd.read_json('res/train.json', typ = 'frame',encoding='utf-8')
        self.val = pd.read_json("res/val.json", typ='frame', encoding = 'utf-8')
        self.test = pd.read_json("res/test.json", typ='frame', encoding = 'utf-8')
        self.data = pd.concat([self.train, self.val, self.test])
        self.song_meta = pd.read_json('res/song_meta.json', typ = 'frame', encoding='utf-8')
        self.plylst_meta = pd.DataFrame(self.train[['id','tags','plylst_title']])
        self.gnr_meta = pd.read_json('res/genre_gn_all.json', typ='series', encoding='utf-8')
        self.minmax = MinMaxScaler()
        ## 가수가 몇명인지 (1. 가수 카운트 (art_cnt))
        self.for_artist_num = self.song_meta[['id', 'artist_id_basket']]['artist_id_basket'].to_dict()
        ## 앨범이 몇개인지 (1. 앨범 카운트 (alb_cnt))
        self.for_album_num = self.song_meta[['id', 'album_id']]['album_id'].to_dict()
        ## genre 파악하기 ( 1. 장르 카운트 (gnr_cnt), 2. 장르 별 비율 gnr_prob)  
        self.for_genre_map = self.song_meta['song_gn_gnr_basket'].to_dict()
        
    def __call__(self, train = False):
        
        if train == True:
            self.data = self.train
        else:
            pass
        self.data['artist_num'] = self.data['songs'].apply(self.artist_length)
        self.data['album_cnt'] = self.data['songs'].apply(self.album_cnt)
        self.data = self.song_ratio(self.data)
        self.data['genre_cnt'] = self.data['songs'].apply(self.genre_cnt)
        self.data = self.data.reset_index()
        gnr_prob = self.data['songs'].apply(self.genre_prob)
        gnr_prob = pd.DataFrame(gnr_prob.tolist(), columns=['gnr_{}'.format(idx) for idx in range(0,30)])
        self.data = pd.concat([self.data, gnr_prob], axis=1)
        self.data = self.date()
        self.data = self.minmax_()

        self.save()
        return self.data

    def artist_length(self, data):
        artists = [artist for song_id in data for artist in self.for_artist_num[song_id]]
        length = list(set(artists)).__len__()

        return length

    def album_cnt(self, data):
        album_ids = [self.for_album_num[song_id] for song_id in data]
        length = list(set(album_ids)).__len__()

        return length

    def song_ratio(self, train):
        ## 앨범 개수 곡 비율
        train['song_cnt'] = train['songs'].map(len)    
        train['album_per_song'] = train['album_cnt']/train['song_cnt']
        ## 아티스트 명수 비율    
        train['artists_per_song'] = train['artist_num']/train['song_cnt']

        train = train.fillna(0)
        '''
        train['album_per_song'].replace(np.inf, 0)
        train['artists_per_song'].replace(np.inf, 0)
        '''

        del train['song_cnt']

        return train

    def genre_cnt(self, data):
        genres = [gnr for song_id in data for gnr in self.for_genre_map[song_id]]
        length = list(set(genres)).__len__()

        return length

    def genre_prob(self, data):
        genre_probs = np.zeros(30)
        genre_name = [self.for_genre_map[song_id] for song_id in data]
        for genres in genre_name:
            if len(genres) == 0:
                score = 0
                pass
            else:
                score = 1/len(genres)
            for gnr in genres:
                if gnr == 'GN9000':
                    continue
                genre_probs[int(gnr[2:4])-1] += score

        return genre_probs / len(genre_name)

    def date(self):
        self.data['updat_date'] = pd.to_datetime(self.data['updt_date'], format='%Y-%m-%d %H:%M:%S', errors='raise')

        #data['date'] = data['updt_date'].dt.date         # YYYY-MM-DD(문자)
        self.data['year']  = self.data['updat_date'].dt.year         # 연(4자리숫자)
        self.data['month']  = self.data['updat_date'].dt.month        # 월(숫자)
        self.data['season'] = self.data['updat_date'].dt.quarter

        self.data['season'][self.data['month'].isin([1,2,12])] = 4  # 겨울
        self.data['season'][self.data['month'].isin([3,4,5])] = 1   # 봄
        self.data['season'][self.data['month'].isin([6,7,8])] = 2  # 여름
        self.data['season'][self.data['month'].isin([9,10,11])] = 3  # 가을

        tmp = pd.DataFrame(columns=['year_section'])
        self.data = pd.concat([self.data,tmp])

        self.data['year_section'][self.data['year'].isin([2005,2006,2007,2008,2009,2010,2011,2012])] = 1
        self.data['year_section'][self.data['year'].isin([2013,2014])] = 2
        self.data['year_section'][self.data['year'].isin([2015,2016])] = 3
        self.data['year_section'][self.data['year'].isin([2017,2018])] = 4
        self.data['year_section'][self.data['year'].isin([2019,2020])] = 5

        df_season = pd.get_dummies(self.data['season']).add_prefix('season') 
        df_year = pd.get_dummies(self.data['year_section']).add_prefix('year_section') 
        self.data = pd.concat([self.data,df_season,df_year],axis=1)
        self.data = self.data.rename(columns={'season1.0':'season_1','season2.0':'season_2','season3.0':'season_3','season4.0':'season_4','year_section1.0':'year_1','year_section2.0':'year_2','year_section3.0':'year_3','year_section4.0':'year_4','year_section5.0':'year_5'})

        self.data = self.data.drop(['updt_date','year','month','season','year_section', 'updat_date'\
            , 'index', 'like_cnt', 'plylst_title', 'songs', 'tags'], axis=1)

        self.data = self.data.fillna(0)

        return self.data

    def minmax_(self):
        self.data[['artist_num', 'album_cnt','genre_cnt']] = self.minmax.fit_transform(self.data[['artist_num','album_cnt','genre_cnt']])

        return self.data

    def save(self):
        self.data.to_json('res/meta_add1.json', orient='records',force_ascii=False)

metaa = mk_meta()
# train = pd.read_json("res/train.json", typ = 'frame', encoding = 'utf-8')
# train['updat_date'] = pd.to_datetime(train['updt_date'], format='%Y-%m-%d %H:%M:%S', errors='raise')
# train['updat_date'].dt.year
# train.head()
# train.drop(['tags', 'id'], axis=1)
meta = metaa(train=False)

# train = train.drop(['num_songs','num_tags','plylst_title', 'songs', 'tags', 'like_cnt','index','songs_cnt','tags_cnt'], axis=1)

# train = train.fillna(0)

# train.to_json('meta_add1.json',orient='records',force_ascii=False)

# meta = pd.read_json('meta.json', encoding='utf-8')
# meta


