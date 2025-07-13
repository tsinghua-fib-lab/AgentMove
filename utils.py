import os
import re
import glob
import json
import jsmin
import argparse
import json_repair
import numpy as np

import hashlib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Tuple
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2


from config import EXP_CITIES, PROCESSED_DIR
from token_count import TokenCount


def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    radius = 6371.0
    distance = radius * c
    return distance


def create_dir(dir):
    # if dir does not exist, create it
    if not os.path.exists(dir):
        os.makedirs(dir)


def convert_time(dataset, model, original_time_str):
    # 解析原始时间字符串的格式
    if dataset in['Shanghai']: # for WWW 2019 Shanghai-ISP
        parsed_time = datetime.strptime(original_time_str, "%a %b %d %H:%M:%S %Y")
    else:
        parsed_time = datetime.strptime(original_time_str, "%a %b %d %H:%M:%S %z %Y")
    # 转换为目标格式的字符串
    if model == "GETNext":
        formatted_time_str = parsed_time.strftime("%Y-%m-%d %H:%M:%S")
    elif model == "SNPM":
        formatted_time_str = parsed_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    elif model == "STHM":
        formatted_time_str = parsed_time.strftime("%Y-%m-%dT%H:%M:%S")
    else:
        raise ValueError("Unsupported model type. Supported models are: GETNext, SNPM, STHM.")
    return formatted_time_str


def string_to_md5_hex(s):
    # 创建MD5哈希对象
    hash_object = hashlib.md5()
    # 更新哈希对象，输入需要是bytes类型
    hash_object.update(s.encode('utf-8'))
    # 获取十六进制形式的摘要
    hex_dig = hash_object.hexdigest()
    return hex_dig


def convert_timestamp(dataset, time_str):
    if dataset in['Shanghai']: # for WWW 2019 Shanghai-ISP
        timestamp = datetime.strptime(time_str, "%a %b %d %H:%M:%S %Y")
    else:
        timestamp = datetime.strptime(time_str, "%a %b %d %H:%M:%S %z %Y")
    midnight = timestamp.replace(hour=0, minute=0, second=0)
    total_minutes = (timestamp - midnight).total_seconds() / 60
    total_minutes_in_day = 24 * 60

    fraction = total_minutes / total_minutes_in_day

    return fraction


def replace_original_poi_id(fs):
    fs['temp_id'] = fs.groupby(['Latitude', 'Longitude','PoiCategoryId']).ngroup() + 1

    # 更新 PoiId，使用 temp_id 作为新的 PoiId
    fs['PoiId'] = fs['temp_id']

    # 删除临时列
    fs.drop(columns='temp_id', inplace=True)

    return fs


def id_encode(fit_df: pd.DataFrame, encode_df: pd.DataFrame, column: str, padding: int = -1) -> Tuple[dict, int]:
    id_le = LabelEncoder()
    id_le = id_le.fit(fit_df[column].values.tolist())
    if padding == 0:
        padding_id = padding
        encode_df[column] = [
            id_le.transform([i])[0] + 1 if i in id_le.classes_ else padding_id
            for i in encode_df[column].values.tolist()
        ]
    else:
        padding_id = len(id_le.classes_)
        encode_df[column] = [
            id_le.transform([i])[0] if i in id_le.classes_ else padding_id
            for i in encode_df[column].values.tolist()
        ]
    return id_le, padding_id    


def ignore_first(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ignore the first check-in sample of every trajectory because of no historical check-in.
    """
    df['pseudo_session_trajectory_rank'] = df.groupby(
        'pseudo_session_trajectory_id')['UTCTimeOffset'].rank(method='first')
    df['query_pseudo_session_trajectory_id'] = df['pseudo_session_trajectory_id'].shift()
    df.loc[df['pseudo_session_trajectory_rank'] == 1, 'query_pseudo_session_trajectory_id'] = None
    df['last_checkin_epoch_time'] = df['UTCTimeOffsetEpoch'].shift()
    df.loc[df['pseudo_session_trajectory_rank'] == 1, 'last_checkin_epoch_time'] = None
    df.loc[df['UserRank'] == 1, 'SplitTag'] = 'ignore'
    df.loc[df['pseudo_session_trajectory_rank'] == 1, 'SplitTag'] = 'ignore'
    return df


def encode_poi_catid(
        fit_df: pd.DataFrame,
        encode_df: pd.DataFrame,
        source_column: str,
        target_column: str,
        padding: int = -1
) -> Tuple[LabelEncoder, int]:
    """
    将source_column列中的唯一值编码到target_column列，类似于STPM的id_encode函数。
    :param fit_df: 用于构建LabelEncoder的DataFrame
    :param encode_df: 需要编码的DataFrame
    :param source_column: 要编码的源列
    :param target_column: 编码后的目标列
    :param padding: 当值不存在于LabelEncoder中时的填充值
    :return: LabelEncoder实例和填充值padding_id
    """
    # 初始化LabelEncoder并进行fit
    id_le = LabelEncoder()
    id_le = id_le.fit(fit_df[source_column].values.tolist())

    # 如果padding为0，编码值从1开始
    if padding == 0:
        padding_id = padding
        encode_df[target_column] = [
            id_le.transform([i])[0] + 1 if i in id_le.classes_ else padding_id
            for i in encode_df[source_column].values.tolist()
        ]
    else:
        # 如果padding不是0，默认填充值为最大编码值+1
        padding_id = len(id_le.classes_)
        encode_df[target_column] = [
            id_le.transform([i])[0] if i in id_le.classes_ else padding_id
            for i in encode_df[source_column].values.tolist()
        ]

    return id_le, padding_id


def int_to_days(int_day):
    days_of_week = {0: 'Monday',
                    1: 'Tuesday',
                    2: 'Wednesday',
                    3: 'Thursday',
                    4: 'Friday',
                    5: 'Saturday',
                    6: 'Sunday'}
    return days_of_week.get(int_day, "NA")


def list_predicted_users(folder_path):
    # get the names of all the files in the folder
    files = os.listdir(folder_path)
    # filter out only the files that are .json
    files = [f for f in files if f.endswith('.json')]
    # split file names to get the user id (second last _ is the split)
    users = [f.split('_')[-2] for f in files]
    # remove duplicates
    users = list(set(users))
    return users


def match_prediction(text, prediction_key="prediction"):
    if prediction_key=="prediction":
        match = re.search(r'[Pp]rediction(.*?)[Rr]eason', text, re.DOTALL)
    elif prediction_key=="recommendation":
        match = re.search(r'[Rr]ecommendation(.*?)[Rr]eason', text, re.DOTALL)
    else:
        match = re.search(r'[Pp]rediction(.*?)[Rr]eason', text, re.DOTALL)
    
    # Extract the prediction text between "prediction" and "reason"
    if match:
        prediction_text = match.group(1)
        place_ids = re.findall(r'\b[0-9a-f]{24}\b', prediction_text)
    else:
        place_ids = []
    return place_ids


def token_count(text):
    tc = TokenCount(model_name="gpt-3.5-turbo")
    return tc.num_tokens_from_string(text)


def extract_json(full_text, prediction_key="prediction"):
        # Attempt to load as JSON
        # we can use json_pair to repair invalid JSON https://github.com/mangiucugna/json_repair
        # we can use jsmin to remove comments in JSON https://github.com/tikitu/jsmin/
        if not isinstance(full_text, str):
            output_json = {
                "raw_response": ""
            }
            prediction = ""
            reason = ""
            return output_json, prediction, reason
        json_str = full_text[full_text.find('{'):full_text.rfind('}') + 1]
        if len(json_str)==0:
            json_str = full_text
        
        # remove potential comments in json_str
        try:
            json_str = jsmin.jsmin(json_str)
        except:
            pass

        try:            
            output_json = json.loads(json_str)
            prediction = output_json.get(prediction_key)
            if len(prediction)==0:
                prediction = match_prediction(output_json, prediction_key)
            reason = output_json.get('reason')
        except json.JSONDecodeError:
            # If not JSON, store the raw full_text string in a new dictionary
            prediction = full_text[full_text.find('['):full_text.rfind(']') + 1]
            reason = ""
            if len(prediction) > 0:
                try:
                    prediction = json.loads(prediction)
                    prediction = [int(item) for item in prediction]
                except:
                    prediction = prediction              
            else:
                prediction = match_prediction(full_text, prediction_key)
            output_json = {
                "raw_response": full_text,
                "prediction": prediction,   
                "reason" : ""       
            }
        except Exception as e:
            reason = "Exception:{}".format(e)
            output_json = {
                "raw_response": full_text,
                "prediction": prediction,   
                "reason" : reason
            }

        return output_json, prediction, reason


def token_analyis(file_path, inlcude=None):
    # for city in ["NewYork", "Tokyo", "Shanghai"]:
    # file_path = f"results/20240803/{city}/agentmove/*"
    print(file_path)
    file_path = os.path.join(glob.glob(file_path)[0], "*")
    print(file_path)
    if inlcude==None:
        file_path = os.path.join(glob.glob(file_path)[0], "*")
    else:
        for file in glob.glob(file_path):
            if inlcude in file:
                file_path = os.path.join(file, "*")
                break
    print(file_path)
    lens = []
    for file in glob.glob(file_path):
        # print(file)
        with open(file) as fid:
            data = json.load(fid)
            input_text_len = token_count(data["input"])
            lens.append(input_text_len)
    res = (file_path, len(lens), np.percentile(lens, 0.5), np.percentile(lens, 0.9), max(lens), np.sum(lens))
    print(res)


def generate_graphs():
    from models.world_model import SocialWorld
    from processing.data import Dataset
    for city_name in EXP_CITIES:
        print("processing {}".format(city_name))
        dataset = Dataset(
            dataset_name=city_name,
            traj_min_len=3,
            trajectory_mode="trajectory_split", 
            historical_stays=16,
            context_stays=6,
            save_dir=PROCESSED_DIR,
            use_int_venue=False,
            )

        social_world = SocialWorld(
            traj_dataset=dataset,
            save_dir=PROCESSED_DIR,
            city_name=city_name,
            khop=1,
            max_neighbors=10
        )


def generate_data():
    from processing.data import Dataset
    for city_name in EXP_CITIES:
        print("processing {}".format(city_name))
        dataset = Dataset(
            dataset_name=city_name,
            traj_min_len=3,
            trajectory_mode="trajectory_split", 
            historical_stays=15,
            context_stays=6,
            save_dir=PROCESSED_DIR,
            use_int_venue=False,
            )

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--file_path', type=str, default="")
    # parser.add_argument('--include', type=str, default="")
    # args = parser.parse_args()

    # token_analyis(args.file_path, args.include)
    
    # generate_graphs()

    generate_data()