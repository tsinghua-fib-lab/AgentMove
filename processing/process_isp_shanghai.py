import pandas as pd 
import numpy as np
import os
from sklearn.neighbors import KDTree

import time
from collections import Counter
from config import WWW2019_DATA_DIR, NO_ADDRESS_TRAJ_DIR

def strings_to_categorical_codes(strings):
    return pd.Categorical(strings).codes

# codes for loading data from private telecom trajectories.
def samples_generator(data_path, data_name, threshold=2000, seed=1):
    tmp = []
    np.random.seed(seed=seed)
    with open(os.path.join(data_path, data_name)) as fid:
        for line in fid:
            user, trace = line.split("\t")
            trace_len = len(trace.split('|'))
            # trace_len = 0
            tmp.append([trace_len, user])
    np.random.shuffle(tmp)
    samples = sorted(tmp, key=lambda x: x[0], reverse=True)
    samples_return = {}
    for u in [x[1] for x in samples[:threshold]]:
        samples_return[u] = len(samples_return)
    return samples_return

def load_cat(data_path, data_name = "poi.txt"):
    vid_list = {}
    vid_lookup = {}
    vid_array = []

    df = pd.read_csv(os.path.join(data_path, data_name), sep=' ' ,header=None, names=[
        "latitude", "longitude", "poi_name", "venue_category_name", "venue2num",
        "venue_num"],encoding='gbk')
    print(df.head())
    df['poi_id'] = strings_to_categorical_codes(df["poi_name"].tolist())
    print(df.head())
    for idx, row in df.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]
        cat = row["venue_category_name"]
        pid = row['poi_id']
        name = row['poi_name']
        if pid not in vid_list:
            cid = len(vid_list) + 1
            vid_list[pid] = [cid, (lat, lon), cat, name]
            vid_lookup[cid] = [pid, (lat, lon)]
            vid_array.append((lat, lon))
    vid_array = np.array(vid_array)
    kdtree_cat = KDTree(vid_array)
    return vid_list, vid_lookup, kdtree_cat  #dist, ind = kdtree.query([(lat, lon)], k=1) bid = vid_lookup[ind[0][0] + 1][0]

def load_data_match_sparse_cat(data_path, data_name, sample_users):
    vid_list, vid_lookup, kdtree_cat = load_cat(data_path)
    #######################
    # default settings
    hour_gap = 24  # 24
    session_max = 20  # 20
    #######################
    filter_short_session = 3
    sessions_count_min = 1
    data = {}
    with open(os.path.join(data_path, data_name)) as fid:
        for line in fid:
            user, traces = line.strip("\r\n").split("\t")
            if user not in sample_users:
                continue
            sessions = {}
            for i, tr in enumerate(traces.split('|')):
                points = tr.split(",")
                if len(points) > 1:
                    if len(points) == 3:
                        tim, _, lon_lat = points   #15,2019001534,121.44788361_31.0318203
                    elif len(points) == 2:
                        tim, lon_lat = points
                    lat, lon = [float(x) for x in lon_lat.split("_")]
                    dist, ind = kdtree_cat.query([(lat, lon)], k=1)
                    pid = vid_lookup[ind[0][0] + 1][0]
                else:
                    continue
                if pid in vid_list:
                    vid = vid_list[pid][0]
                    tid = int(tim)
                    day = int(tid / 24)
                    cat = vid_list[pid][2]
                    name = vid_list[pid][3]
                else:
                    continue
                record = [day, vid, tid % 24, cat, name, (lon, lat)]
                sid = len(sessions)
                if i == 0 or len(sessions) == 0:
                    sessions[sid] = [record]
                else:
                    if (tid - last_tid) > hour_gap or len(sessions[sid - 1]) > session_max:
                        sessions[sid] = [record]
                    else:
                        sessions[sid - 1].append(record)
                last_tid = tid
                sessions_filter = {}
                for s in sessions:
                    if len(sessions[s]) >= filter_short_session:
                        sessions_filter[len(sessions_filter)] = sessions[s]
            if len(sessions_filter) >= sessions_count_min:
                data[user] = {"sessions": sessions_filter}
    all_rows = []
    start_time = "Tue Apr 19 00:00:00 2016"
    start_time = time.mktime(time.strptime(start_time,"%a %b %d %H:%M:%S %Y"))
    for user, sessions in data.items():
        for _, session in sessions.items():
            for si, traj_points in session.items():
                if COMPRESS:
                    traj_points = dense_session_compress(traj_points)
                for traj_point in traj_points:
                    real_time = (traj_point[0]*24 + traj_point[2])*3600+start_time
                    real_time_str = time.asctime( time.localtime(real_time) )
                    all_rows.append({"city":"Shanghai_Weibo", "user_id":user, "traj_id": si, "utc_time": real_time_str,"venue_id":traj_point[1],"venue_name": traj_point[4],'longitude':traj_point[5][0],'latitude':traj_point[5][1],"venue_category_name":traj_point[3]})
    result = pd.DataFrame(all_rows)
    return result


def dense_session_compress(original_trace):
    # TODO: merge and select the location during the same half an hour
    compress_time_threshold = 120

    vid_list_local = {}
    for i, x in enumerate(Counter([p[1] for p in original_trace]).most_common()):
        vid_list_local[x[0]] = i + 1000

    transferred_trace = [[vid_list_local[p[1]], (p[0]*24+p[2])*60] for p in original_trace]

    # find most common location index in the same half an hour
    info = np.array(transferred_trace)
    index = np.floor(info[:, 1] / compress_time_threshold)
    index_count = Counter(index).most_common()
    select = [x[0] for x in index_count if x[1] > 1]
    select_des = []
    for s in range(len(select)):
        find = np.argwhere(index == select[s])
        candidate = np.squeeze(info[find, 0], 1)
        des = Counter(candidate).most_common()[0][0]
        select_des.append(des)

    # replace the location index in records with the most common location
    day_trace = []
    for i, id in enumerate(index):
        if id not in select:
            day_trace.append([int(transferred_trace[i][0]), i])
        else:
            sid = select.index(id)
            if i == int(np.argwhere(index == select[sid])[0][0]):
                des = select_des[sid]
                day_trace.append([int(des), i])

    # delete contiguous location repetition
    compress_trace = [original_trace[0]]
    last_des = day_trace[0][0]
    for r in day_trace:
        if r[0] == last_des:
            continue
        else:
            last_des = r[0]
            _, ind = r
            compress_trace.append(original_trace[ind])
    return compress_trace


def load_data_match_cat_telecom(data_path, data_name, sample_users=None):
    ##################
    filter_short_session = 3
    sessions_count_min = 3
    day_start=8
    day_end=20
    ##################
    vid_list, vid_lookup, kdtree_cat = load_cat(data_path)
    data = {}
    #3360862	|15,2019001534,121.44788361_31.0318203|   user,time,vid,lon_lat
    with open(os.path.join(data_path, data_name)) as fid:  #isp.txt
        for line in fid:
            user, traces = line.strip("\r\n").split("\t")
            if sample_users is not None:
                if user not in sample_users:
                    continue
            sessions = {}
            for tr in traces.split('|'):
                points = tr.split(",")
                if len(points) > 1:
                    if len(points) > 2:
                        tim,_, lon_lat = points
                    elif len(points) == 2:
                        tim, lon_lat = points
                    else:
                        tim,_, lon_lat = points
                    lon, lat = [float(x) for x in lon_lat.split("_")]
                    dist, ind = kdtree_cat.query([(lat, lon)], k=1)
                    pid = vid_lookup[ind[0][0] + 1][0]

                    if int(tim)%24<day_start or int(tim)%24>day_end:
                        continue

                    if pid in vid_list:
                        vid = vid_list[pid][0]  #cid
                        tim = int(tim)
                        day = int(tim / 24)
                        cat = vid_list[pid][2]
                        name = vid_list[pid][3]
                        if day not in sessions:
                            sessions[day] = [[day, vid, tim % 24, cat, name, (lon, lat)]]
                        else:
                            sessions[day].append([day, vid, tim % 24, cat, name, (lon, lat)])
            #"CapeTown_4c9106162626a1cddbbb2e6b": {"administrative": "City of Cape Town", "subdistrict": "Cape Town Ward 73", "poi": "Shell", "street": "Main Road"}
            #city,user,time,venue_id,utc_time,lon,lat,venue_cat_name
            # Beijing,83132,480,4d67ecb5052ea1cd2b5aa049,Tue Apr 03 18:28:06 +0000 2012,116.437258,39.918656,Lounge
            sessions_filter = {}
            for s in sessions:
                if len(sessions[s]) >= filter_short_session:
                    sessions_filter[len(sessions_filter)] = sessions[s]
            if len(sessions_filter) >= sessions_count_min:
                data[user] = {"sessions": sessions_filter}
        print("telecom users:{}".format(len(data.keys())))
    # city,user,time,venue_id,utc_time,lon,lat,venue_cat_name
    start_time = "Tue Apr 19 00:00:00 2016"
    start_time = time.mktime(time.strptime(start_time,"%a %b %d %H:%M:%S %Y"))
    all_rows = []
    for user, sessions in data.items():
        for _, session in sessions.items():
            for si, traj_points in session.items():
                if COMPRESS:
                    traj_points = dense_session_compress(traj_points)
                for traj_point in traj_points:
                    real_time = (traj_point[0]*24 + traj_point[2])*3600+start_time
                    real_time_str = time.asctime( time.localtime(real_time) )
                    all_rows.append({"city":"Shanghai_ISP", "user_id":user, "traj_id": si, "utc_time": real_time_str,"venue_id":traj_point[1],"venue_name": traj_point[4],'longitude':traj_point[5][0],'latitude':traj_point[5][1],"venue_category_name":traj_point[3]})
    result = pd.DataFrame(all_rows)
    return result

if __name__ == '__main__':
    COMPRESS = True
    sample_users = samples_generator(WWW2019_DATA_DIR, "weibo", threshold=2000)
    data_dense_cat = load_data_match_cat_telecom(WWW2019_DATA_DIR, 'isp', sample_users=sample_users)
    data_sparse_cat =  load_data_match_sparse_cat(WWW2019_DATA_DIR, 'weibo', sample_users=sample_users)
    data_dense_cat.to_csv(os.path.join(NO_ADDRESS_TRAJ_DIR, "Shanghai_ISP_filtered.csv"), index=False)
    data_sparse_cat.to_csv(os.path.join(NO_ADDRESS_TRAJ_DIR,"Shanghai_Weibo_filtered.csv"), index=False)