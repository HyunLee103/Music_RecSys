from experiments import song_valid
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from arena_util import load_json
from collections import Counter

args = {
    'meta_len': 10,
    'meta_d': 8,
    'songs_len': 10000,
    'ply_d': 8
}

class PlyDataset(Dataset):
    def __init__(self, ply, meta):
        super().__init__()
        self.ply = ply
        self.meta = meta
        
    def __getitem__(self, index):
        return {'ply':self.ply[index], 'meta':self.meta[index]}
    
    def __len__(self):
        return len(self.ply)

class DeepMF(nn.Module):
    def __init__(self, args):
        super(DeepMF, self).__init__()
        
        self.meta_len = args["meta_len"]
        self.meta_d = args["meta_d"]
        self.songs_len = args['songs_len']
        self.ply_d = args['ply_d']
        self.activation = nn.ReLU()
        
        self.meta_embedding = nn.Linear(self.meta_len,
                                        self.meta_d)
        self.ply_embedding = nn.Linear(self.songs_len,
                                       self.ply_d)
        
        self.decoder = nn.Linear(self.meta_d + self.ply_d,
                                 self.song_len)
        
    def forward(self, ply, meta):
        ply_embed = self.activation(self.ply_embedding(ply))
        meta_embed = self.activation(self.meta_embedding(meta))
        latent = torch.cat((ply_embed, meta_embed), dim=-1)
        ply_recon = self.decoder(latent)
        
        return torch.sigmoid(ply_recon)
    
# train = load_json("arena_data/orig/train.json")
# val = load_json("arena_data/questions/val.json")
train = load_json("res/train.json")
val = load_json("res/val.json")

data = train + val

# count train songs. filter under 150
def song_count_filter(data, over_n):
    counter = Counter()

    for ply in data:
        counter.update(ply['songs'])

    song_valid = set([song_id for song_id, cnt in counter.items() if cnt >= over_n])
    print(f"song_count_filter\n- song_valid length: {len(song_valid)}")
    
    return song_valid
    
song_valid = song_count_filter(train, over_n=50)

# remove invaild song.
for ply in train:
    ply['songs'] = [song for song in ply['songs'] if song in song_valid]

cnt = 0
for ply in real_train:
    if not len(ply['songs']):
        cnt += 1
train_empty_cnt
len(real_train)

for ply in real_val:
    ply['songs'] = [song for song in ply['songs'] if song in song_valid]        

cnt = 0
for ply in real_val:
    if not len(ply['songs']):
        cnt += 1
origin_empty_cnt
cnt
len(real_val)
cnt - origin_empty_cnt
len(train)

lr = 1e-3
n_epochs = 1000
batch_size = 16

model = DeepMF(args)


loss = nn.BCELoss()
optim = torch.optim.Adam(model.parameters(), lr=lr)

