from arena_util import load_json, write_json
import pandas as pd
import matplotlib.pyplot as plt

res_val = load_json("res/val.json")
res_val = pd.DataFrame(res_val)

res_val['ply_len'] = res_val['songs'].apply(len)

res_val[(res_val['ply_len'] > 0) & (res_val['ply_len'] < 4)]

res_val['ply_len'] > 0

len(res_val)

res_val['ply_len'].d()

plt.hist(res_val[res_val['ply_len'] != 0]['ply_len'])


