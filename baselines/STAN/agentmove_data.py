import os
import json
import pandas as pd
import numpy as np
import argparse
from email.utils import parsedate_to_datetime
from .config import ORIGIN_TRAJ_DATA, SAVE_DIR, MAX_TRAJ_LEN, PROMPT_NUM

parser = argparse.ArgumentParser()
parser.add_argument('--city_name', type=str, default='Beijing')
parser.add_argument('--traj_min_len', type=int, default=3)
parser.add_argument('--traj_max_len', type=int, default=50)
args = parser.parse_args()


agentmove_data = json.load(open(os.path.join(ORIGIN_TRAJ_DATA, "align_locations_{}_trajectory_split.json".format(args.city_name))))

# 待测试用户和轨迹
test_user_ids = {}
# 轨迹数据集合
traj_data = []

user_list = [str(y) for y in sorted([int(x) for x in list(agentmove_data.keys())])]
count = 0
for i, user_id in enumerate(user_list):
    v = agentmove_data[user_id]
    traj_ids = [str(y) for y in sorted([int(x) for x in list(v.keys())])]
    if len(traj_ids)==0:
        continue
    test_traj_id = traj_ids[0]

    cur_trajs = v[test_traj_id]["historical_stays_long"] + v[test_traj_id]['context_stays']
    cur_trajs_pd = pd.DataFrame(data=cur_trajs, columns=["city", "user_id", "venue_id", "utc_time", "longitude", "latitude", "venue_category_name", "venue_id_int"])
    # cur_trajs_pd["test_traj_id"] = test_traj_id

    # 按照轨迹数量筛选测试用户
    if len(traj_ids) < args.traj_min_len:
        continue
    if len(traj_ids) > args.traj_max_len:
        continue
    count += 1

    if count > PROMPT_NUM:
        continue
    # STAN逻辑，导致部分比较短的轨迹无法参与训练，可能导致对不齐
    if cur_trajs_pd.shape[0]<MAX_TRAJ_LEN+5:
        continue
    # 测试轨迹ID，用户ID在排序后的轨迹中序号，测试轨迹长度
    test_user_ids[user_id] = (test_traj_id, i, len(v[test_traj_id]['context_stays']))

    traj_data.append(cur_trajs_pd)

# 轨迹数据集合
traj_data_pd = pd.concat(traj_data)

poi = traj_data_pd.filter(items=['venue_id_int', 'latitude', 'longitude'])
poi = poi.groupby('venue_id_int').mean().reset_index()
poi['geo_id'] = poi.index
poi_id_mapping = poi.set_index("venue_id_int")["geo_id"].to_dict()
poi_info = poi[['geo_id','latitude', 'longitude']].to_numpy()


# traj_data_pd['user_id_int'] = pd.factorize(traj_data_pd['user_id'])[0]+1
user = pd.unique(traj_data_pd['user_id'])
user = pd.DataFrame(user, columns=['user_id'])
user['user_id_int'] = user.index  #为UserID重编号
user["user_id_int"] = user["user_id_int"] + 1
traj_user_id_mapping = user.set_index("user_id")['user_id_int'].to_dict()
# 待测试用户和轨迹， 用户ID为新的映射
test_user_ids_dict = {}
for user_id in test_user_ids:
    user_id_int = traj_user_id_mapping[int(user_id)]
    test_user_ids_dict[user_id_int] = test_user_ids[user_id]


# 轨迹数据列重命名和排序
traj_data_pd['user_id_int'] = traj_data_pd["user_id"].apply(lambda x: traj_user_id_mapping[x])
traj_data_pd["time"] = traj_data_pd["utc_time"].apply(parsedate_to_datetime)
traj_data_pd["geo_id"] = traj_data_pd["venue_id_int"].apply(lambda x: poi_id_mapping[x])
traj_data_pd = traj_data_pd.rename(columns={"geo_id": "location", "user_id_int": "entity_id"})
traj_data_pd = traj_data_pd.sort_values(by=["entity_id", "time"])
traj_data_pd_min = traj_data_pd[['time', 'entity_id', 'location']]

start_times = traj_data_pd_min.groupby('entity_id')['time'].transform('min')
# 计算时间差，以分钟为单位
traj_data_pd_min['time_diff'] = (traj_data_pd_min['time'] - start_times).dt.total_seconds() / 60
# 选择需要的列
traj_data_info = traj_data_pd_min[['entity_id', 'location', 'time_diff']].astype(np.int32).to_numpy()

np.save(os.path.join(SAVE_DIR, f"{args.city_name}.npy"), traj_data_info)
np.save(os.path.join(SAVE_DIR, f"{args.city_name}_POI.npy"), poi_info)
with open(os.path.join(SAVE_DIR, f"{args.city_name}_testing_set.json"), "w") as wid:
    json.dump(test_user_ids_dict, wid)

print("our data:")
print("traj point  data shape:", traj_data_info.shape,"poi data shape:",poi_info.shape)
print("number of users:",len(set(traj_data_info[:,0])),"number of locations:",len(set(traj_data_info[:,1])))