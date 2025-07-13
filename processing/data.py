import pandas as pd
import numpy as np
import os
import glob
import requests
import tqdm
from sklearn.model_selection import train_test_split
import json
from datetime import datetime, timedelta

import utils
from config import DATASET, CITY_DATA_DIR, OFFSET_DICT, PROCESSED_DIR
import pickle
import random
import argparse


class Dataset:

    def __init__(self, base_name="AgentMove", dataset_name='nyc', trajectory_mode='trajectory_split', historical_stays=40, context_stays=6,
                 hour_bins=72, traj_min_len=5, traj_max_len=100, max_sample_trajectories=100, save_dir='data/processed', use_int_venue=False, 
                 train_sample=0.7, test_size=0.2, test_sample=200):
        self.dataset_name = dataset_name
        self.trajectory_mode = trajectory_mode
        self.history_stays = historical_stays
        self.context_stays = context_stays
        self.save_dir = save_dir
        self.hour_bins = hour_bins
        self.traj_min_len = traj_min_len
        self.traj_max_len = traj_max_len
        self.max_sample_trajectories = max_sample_trajectories
        # default data processing for LLM based methods, all llm methods use this AgentMove, other names are used for DL methods
        self.base_name = base_name

        self.test_dictionary = {}
        self.true_locations = {}
        self.align_dictionary = {}
        self.processed_datasets = {}
        self.use_int_venue = use_int_venue

        self.sample_one_traj_of_user = True
        # align with the sampling users for agentmove which only test 'prompt_num' users and default value is 200
        self.prompt_num = test_sample
        self.train_sample = train_sample
        self.test_size = test_size
        self.seed = 1234

        if self.trajectory_mode not in ['user_split', 'trajectory_split']:
            raise ValueError('Invalid trajectory mode! Please use either user_split or trajectory_split')

        self.get_processed_datasets()

        if self.dataset_name not in self.processed_datasets:
            print('Loading the dataset...')
            self.data = self.get_dataset()

            print('Computing trajectories...')
            self.get_trajectories()
        else:
            self.test_dictionary = json.load(open(os.path.join(self.save_dir, self.processed_datasets[dataset_name]["test"])))
            self.true_locations = json.load(open(os.path.join(self.save_dir, self.processed_datasets[dataset_name]["true"])))


    def get_encode(self, df):
        """便于DL训练，对过滤后的轨迹重编码"""
        if self.base_name == "STHM":
            df['Latitude'] = df['Latitude'].round(3)
            df['Longitude'] = df['Longitude'].round(3)
            df['UserRank'] = df.groupby('UserId')['UTCTimeOffset'].rank(method='first')
            df = df.sort_values(by=['UserId', 'UTCTimeOffset'], ascending=True)
            # do label encoding
            df_train = df[df['SplitTag'] == 'train']
            # padding id use 0, 
            poi_id_le, padding_poi_ie = utils.id_encode(df_train, df, 'PoiId',padding=-1) 
            category_code_le, padding_category_code = utils.encode_poi_catid(df_train, df,'PoiCategoryCode','PoiCategoryId',padding=-1) 
            poi_category_le, padding_poi_category = utils.id_encode(df_train, df, 'PoiCategoryId',padding=-1) 
            user_id_le, padding_user_id = utils.id_encode(df_train, df, 'UserId',padding=-1) 
            hour_id_le, padding_hour_id = utils.id_encode(df_train, df, 'UTCTimeOffsetHour',padding=-1)
            weekday_id_le, padding_weekday_id = utils.id_encode(df_train, df, 'UTCTimeOffsetWeekday',padding=-1)
            # os.path.join(f"baselines/Spatio-Temporal-Hypergraph-Model/dataset/{self.train_sample}", self.dataset_name, "preprocessed")
            with open(os.path.join(f"baselines/Spatio-Temporal-Hypergraph-Model/dataset/{str(self.train_sample)}", self.dataset_name, "preprocessed",'label_encoding.pkl'), 'wb') as f:
                pickle.dump([
                    poi_id_le, poi_category_le, user_id_le, hour_id_le, weekday_id_le,
                    padding_poi_ie, padding_poi_category, padding_user_id, padding_hour_id, padding_weekday_id
                ], f) 

            # Re-encode trajectory IDs and check-in IDs to ensure continuity
            df['check_ins_id'] = df['UTCTimeOffset'].rank(ascending=True, method='first') - 1 
            traj_id_map = {id: idx for idx, id in enumerate(sorted(df['trajectory_id_raw'].unique()))}
            df['trajectory_id'] = df.apply(lambda x: f"{x['UserId']}_{traj_id_map[x['trajectory_id_raw']]}", axis=1)
            traj_id_map = {id: idx for idx, id in enumerate(sorted(df['pseudo_session_trajectory_id'].unique()))}
            df['pseudo_session_trajectory_id'] = df['pseudo_session_trajectory_id'].map(traj_id_map)
            # Ignore the first check-in of every trajectory when creating samples
            df = utils.ignore_first(df)
       # 用训练集编码         
        elif self.base_name == "GETNext":
            df['UserRank'] = df.groupby('user_id')['timezone'].rank(method='first')
            df = df.sort_values(by=['user_id', 'timezone'], ascending=True)
            # do label encoding
            df_train = df[df['SplitTag'] == 'train']
            # padding id use 0
            poi_id_le, padding_poi_ie = utils.id_encode(df_train, df, 'POI_id', padding=-1)
            poi_category_le, padding_poi_category = utils.id_encode(df_train, df, 'POI_catid', padding=-1)
            user_id_le, padding_user_id = utils.id_encode(df_train, df, 'user_id', padding=-1)
            # df['POI_id'],_ = utils.id_encode(df_train, df, 'POI_id', padding=0)
            # df['POI_catid'],_ = utils.id_encode(df_train, df, 'POI_catid', padding=0)
            # df['user_id'],_ = utils.id_encode(df_train, df, 'user_id', padding=0)

            # Re-encode trajectory IDs and check-in IDs to ensure continuity
            df['check_ins_id'] = df['timezone'].rank(ascending=True, method='first') - 1 
            traj_id_map = {id: idx for idx, id in enumerate(sorted(df['trajectory_id_raw'].unique()))}
            df['trajectory_id'] = df.apply(lambda x: f"{x['user_id']}_{traj_id_map[x['trajectory_id_raw']]}", axis=1) 
        elif self.base_name == "SNPM":
            df = df.sort_values(by=['user_id', 'UTC_time'], ascending=True)
            # do label encoding
            df_train = df[df['SplitTag'] == 'train']
            # padding id use 0
            poi_id_le, padding_poi_ie = utils.id_encode(df_train, df, 'POI_id', padding=0)
            user_id_le, padding_user_id = utils.id_encode(df_train, df, 'user_id', padding=0)  
        return df
    

     # 训练集是否要选择仅出现在测试集的用户  ？     
    def get_baseline(self, user_ids, DL_train_trajectory_ids, DL_val_trajectory_ids, DL_test_trajectory_ids):
        """用于其他DL baselines训练时对齐数据"""
        output = pd.DataFrame()
        if self.base_name == "STHM": # 用全量数据构图
            # sample_file即为没有过滤的原始训练+验证+测试数据
            preprocessed_path = os.path.join(f"baselines/Spatio-Temporal-Hypergraph-Model/dataset/{self.train_sample}", self.dataset_name, "preprocessed")
            if not os.path.exists(preprocessed_path):
                os.makedirs(preprocessed_path)
            sample_file = os.path.join(preprocessed_path, 'sample.csv')
            train_file = os.path.join(preprocessed_path, 'train_sample.csv')
            validate_file = os.path.join(preprocessed_path, 'validate_sample.csv')
            test_file = os.path.join(preprocessed_path, 'test_sample.csv')
            output['UserId'] = self.data['user_id']
            output['PoiId'] = self.data['venue_id']
            output['PoiCategoryName'] = self.data['venue_category_name']
            output['PoiCategoryId'] = self.data['venue_category_name'].astype('category').cat.codes
            output['PoiCategoryCode'] = self.data['venue_category_name'].apply(lambda x: utils.string_to_md5_hex(x))
            output['Latitude'] = self.data['latitude']
            output['Longitude'] = self.data['longitude']
            output['trajectory_id_raw'] = self.data['DL_traj_id']
            output['pseudo_session_trajectory_id'] = self.data['DL_traj_id']
            output['TimezoneOffset'] = OFFSET_DICT[self.dataset_name]
            # dataset, model, original_time_str
            output['UTCTime'] = self.data['utc_time'].apply(lambda x: utils.convert_time(self.dataset_name, self.base_name, x))
            output['UTCTimeOffset'] = output.apply(
    lambda x: datetime.strptime(x['UTCTime'], "%Y-%m-%dT%H:%M:%S") + timedelta(hours=OFFSET_DICT[self.dataset_name] / 60),
    axis=1)
            # output['UTCTimeOffset'] = output['UTCTimeOffset'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
            output['UTCTimeOffsetEpoch'] = output['UTCTimeOffset'].apply(lambda x: x.strftime('%s'))
            output['UTCTimeOffsetWeekday'] = output['UTCTimeOffset'].apply(lambda x: x.weekday())
            output['UTCTimeOffsetHour'] = output['UTCTimeOffset'].apply(lambda x: x.hour)
            output['UTCTimeOffsetDay'] = output['UTCTimeOffset'].apply(lambda x: x.strftime('%Y-%m-%d'))
            output['UserRank'] = output.groupby('UserId')['UTCTimeOffset'].rank(method='first')
            # output['UTCTimeOffsetNormInDayTime'] = self.data['UTCTimeOffset'].apply(lambda x: utils.convert_timestamp(self.dataset_name, x))  
            output['check_ins_id'] = output['UTCTimeOffset'].rank(ascending=True, method='first') - 1 
            output.loc[(output['UserId'].isin(user_ids)) & (output['pseudo_session_trajectory_id'].isin(set(DL_val_trajectory_ids)|set(DL_train_trajectory_ids)|set(DL_test_trajectory_ids))), 'SplitTag'] = 'all'
            output.loc[(output['UserId'].isin(user_ids)) & (output['pseudo_session_trajectory_id'].isin(DL_train_trajectory_ids)), 'SplitTag'] = 'train'
            output.loc[(output['UserId'].isin(user_ids)) & (output['pseudo_session_trajectory_id'].isin(DL_val_trajectory_ids)), 'SplitTag'] = 'validation'
            output.loc[(output['UserId'].isin(user_ids)) & (output['pseudo_session_trajectory_id'].isin(DL_test_trajectory_ids)), 'SplitTag'] = 'test'
            output = output[output['SplitTag'].isin(['train','test','validation'])]            
            output = self.get_encode(output)

            columns_output = ['check_ins_id','UTCTimeOffset','UTCTimeOffsetEpoch','pseudo_session_trajectory_id','query_pseudo_session_trajectory_id','UserId','Latitude','Longitude','PoiId','PoiCategoryId','PoiCategoryName','last_checkin_epoch_time','UTCTimeOffsetEpoch', 'UTCTimeOffsetWeekday','UTCTimeOffsetHour']
            train_part = output[output['SplitTag'] == 'train']
            val_part = output[output['SplitTag'] == 'validation']
            test_part = output[output['SplitTag'] == 'test']
            # 将筛选并排序后的DataFrame输出到CSV文件
            output[columns_output].to_csv(sample_file, index=False)
            train_part[columns_output].to_csv(train_file, index=False)
            val_part[columns_output].to_csv(validate_file, index=False)
            test_part[columns_output].to_csv(test_file, index=False)

        elif self.base_name == "GETNext":  # 用训练集构图
            preprocessed_path = os.path.join(f"baselines/GETNext/dataset/{str(self.train_sample)}", self.dataset_name)
            if not os.path.exists(preprocessed_path):
                os.makedirs(preprocessed_path)
            sample_file = os.path.join(preprocessed_path, 'sample.csv')
            train_file = os.path.join(preprocessed_path, 'train.csv')
            validate_file = os.path.join(preprocessed_path, 'val.csv')
            test_file = os.path.join(preprocessed_path, 'test.csv')
            output['user_id'] = self.data['user_id']
            output['POI_id'] = self.data['venue_id']
            output['POI_catname'] = self.data['venue_category_name']
            output['POI_catid'] = self.data['venue_category_name'].astype('category').cat.codes
            output['latitude'] = self.data['latitude']
            output['longitude'] = self.data['longitude']
            output['trajectory_id_raw'] = self.data['DL_traj_id']
            output['TimezoneOffset'] = OFFSET_DICT[self.dataset_name]
            output['UTC_time'] = self.data['utc_time'].apply(lambda x: utils.convert_time(self.dataset_name, self.base_name, x))
            output['timezone'] = output.apply(
    lambda x: datetime.strptime(x['UTC_time'], "%Y-%m-%d %H:%M:%S") + timedelta(hours=OFFSET_DICT[self.dataset_name] / 60),
    axis=1)
            output['day_of_week'] = output['UTC_time'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").weekday())
            output['POI_catid_code'] = self.data['venue_category_name'].apply(lambda x: utils.string_to_md5_hex(x))
            output['norm_in_day_time'] = self.data['utc_time'].apply(lambda x: utils.convert_timestamp(self.dataset_name, x))  
            output.loc[(output['user_id'].isin(user_ids)) & (output['trajectory_id_raw'].isin(set(DL_val_trajectory_ids)|set(DL_train_trajectory_ids)|set(DL_test_trajectory_ids))), 'SplitTag'] = 'all'
            output.loc[(output['user_id'].isin(user_ids)) & (output['trajectory_id_raw'].isin(DL_train_trajectory_ids)), 'SplitTag'] = 'train'
            output.loc[(output['user_id'].isin(user_ids)) & (output['trajectory_id_raw'].isin(DL_val_trajectory_ids)), 'SplitTag'] = 'validation'
            output.loc[(output['user_id'].isin(user_ids)) & (output['trajectory_id_raw'].isin(DL_test_trajectory_ids)), 'SplitTag'] = 'test'
            # 先划分之后，用训练集对其他进行编码
            output = self.get_encode(output)
            output.to_csv(sample_file, index=False)
            output[output['SplitTag'] == 'train'].to_csv(train_file, index=False)
            output[output['SplitTag'] == 'validation'].to_csv(validate_file, index=False)
            output[output['SplitTag'] == 'test'].to_csv(test_file, index=False)
        elif self.base_name == "SNPM":
            # user,time, lat,lon ,location
            desired_columns = ['user_id', 'UTC_time', 'latitude', 'longitude', 'POI_id']
            preprocessed_path = os.path.join("baselines/SNPM/data", self.dataset_name)
            if not os.path.exists(preprocessed_path):
                os.makedirs(preprocessed_path)
            sample_file = os.path.join(preprocessed_path, f'checkins-{self.dataset_name}.txt')
            dict_file = os.path.join(preprocessed_path, f'checkins-{self.dataset_name}-dict.json')
            output['user_id'] = self.data['user_id']
            output['POI_id'] = self.data['venue_id']
            output['latitude'] = self.data['latitude']
            output['longitude'] = self.data['longitude']
            output['trajectory_id_raw'] = self.data['DL_traj_id']
            output['UTC_time'] = self.data['utc_time'].apply(lambda x: utils.convert_time(self.dataset_name, self.base_name, x))
            output.loc[(output['user_id'].isin(user_ids)) & (output['trajectory_id_raw'].isin(set(DL_val_trajectory_ids)|set(DL_train_trajectory_ids)|set(DL_test_trajectory_ids))), 'SplitTag'] = 'all'
            output.loc[(output['user_id'].isin(user_ids)) & (output['trajectory_id_raw'].isin(DL_train_trajectory_ids)), 'SplitTag'] = 'train'
            output.loc[(output['user_id'].isin(user_ids)) & (output['trajectory_id_raw'].isin(DL_val_trajectory_ids)), 'SplitTag'] = 'validation'
            output.loc[(output['user_id'].isin(user_ids)) & (output['trajectory_id_raw'].isin(DL_test_trajectory_ids)), 'SplitTag'] = 'test'
            output = self.get_encode(output)
            # 统计每个用户的 SplitTag
            split_counts = output.groupby('user_id')['SplitTag'].value_counts().unstack(fill_value=0)

            # Output train/val/test data for baseline
            user_split_counts = {
                int(user_id): [int(row.get('train', 0)) + int(row.get('validation', 0)),int(row.get('test', 0))]
                for user_id, row in split_counts.iterrows()
            }
            output.to_csv(sample_file, index=False, sep='\t', columns=desired_columns, header=False)
            with open(dict_file, 'w', encoding="utf-8") as file:
                json.dump(user_split_counts, file, ensure_ascii=False)


    def get_processed_datasets(self):
        for x in glob.glob(os.path.join(self.save_dir, "*")):
            file_name = x.split(os.sep)[-1]
            if "test" in file_name or "true" in file_name:
                if self.use_int_venue and ("int" not in file_name):
                    continue
                city_name = file_name.split("_")[2]
                file_type = file_name.split("_")[0]
                if city_name in self.processed_datasets:
                    self.processed_datasets[city_name][file_type] = file_name
                else:
                    self.processed_datasets[city_name] = {file_type: file_name}

    def get_generated_datasets(self):
        return self.test_dictionary, self.true_locations


    def get_dataset(self):
        """
        A function to load the dataset into a pandas dataframe
        :return:
        """
        if self.dataset_name == 'nyc':
            data = pd.read_csv('data/dataset_tsmc2014/dataset_TSMC2014_NYC.txt', delimiter="\t", header=None,
                                   names=["user_id", "venue_id", "venue_category", "venue_category_name", "latitude",
                                          "longitude", "timezone_offset", "utc_time"], encoding='ISO-8859-1')
        elif self.dataset_name == 'tky':
            data = pd.read_csv('data/dataset_tsmc2014/dataset_TSMC2014_TKY.txt', delimiter="\t", header=None,
                               names=["user_id", "venue_id", "venue_category", "venue_category_name", "latitude",
                                      "longitude", "timezone_offset", "utc_time"], encoding='ISO-8859-1')
        else:
            file_path = os.path.join(CITY_DATA_DIR,'{}_filtered.csv'.format(self.dataset_name))
            
            data = pd.read_csv(file_path, header=0, encoding='utf-8')
            rename_dict = {
                'user': 'user_id',
                'time': 'timezone_offset',
                'venue_cat_name': 'venue_category_name',
            }
            data.rename(columns=rename_dict, inplace=True)


        #city,user,time,venue_id,utc_time,lon,lat,venue_cat_name,admin,subdistrict,poi,street
        # convert date time to pandas datetime
        data['datetime'] = pd.to_datetime(data['utc_time'])
        # create a column for the day of the week
        data['weekday'] = data['datetime'].dt.dayofweek
        # convert the weekday to a string
        data['weekday'] = data['weekday'].apply(lambda x: utils.int_to_days(x))
        # create a column for the hour of the day
        data['hour'] = data['datetime'].dt.hour
        # create a column with 'AM' if the hour is less than 12, else 'PM'
        data['am_pm'] = np.where(data['hour'] < 12, 'AM', 'PM')
        # if column 'hour' is higher than 12, subtract 12 from it
        data['hour'] = np.where(data['hour'] > 12, data['hour'] - 12, data['hour'])
        # concatenate the hour and am_pm columns
        data['hour'] = data['hour'].astype(str) + ' ' + data['am_pm']
        # Map 'venue_id' to unique integer IDs
        data['venue_id_int'] = pd.factorize(data['venue_id'])[0]
        
        data['venue_category_name'] = data['venue_category_name'].replace(['NaN', 'NAN', 'nan', None, np.nan, '', 'None', 'none', 'NULL', 'N/A', 'n/a', 'missing', ' ', -999, 999999], '')

        df_grouped = data.groupby('venue_id').agg({
            'latitude': 'mean',
            'longitude': 'mean'
        }).reset_index()

        # merge the grouped dataframe with the original dataframe
        df_merged = pd.merge(data, df_grouped, on='venue_id', suffixes=('', '_mean'))

        # drop the original 'Latitude' and 'Longitude' columns
        df_merged = df_merged.drop(columns=['latitude', 'longitude'])

        # rename the '_mean' columns
        df_merged = df_merged.rename(columns={'latitude_mean': 'latitude', 'longitude_mean': 'longitude'})
        return df_merged


    def test_traj_sampling(self, user_ids):
        counter = 0
        test_trajectory_ids = []
        for user_id in user_ids:
            traj_ids = self.data[self.data['user_id'] == user_id]['DL_traj_id'].values

            # 按照轨迹数量筛选测试用户
            if self.dataset_name in ["Shanghai_Weibo", "Shanghai_ISP"]:
                if len(traj_ids)==0:
                    continue
            else:
                if len(set(self.DL_test_trajectory_ids)&set(traj_ids)) < 3:
                    continue
                if len(set(self.DL_test_trajectory_ids)&set(traj_ids)) > self.traj_max_len:
                    continue

            traj_count=0
            sorted_test_traj = sorted(list(set(self.DL_test_trajectory_ids)&set(traj_ids)))
            for traj_id in sorted_test_traj:
                test_trajectory_ids.append(traj_id)
                counter += 1
                traj_count += 1
                if self.sample_one_traj_of_user:
                    break
                else:
                    if traj_count>self.max_sample_trajectories:
                        break
            
            # TODO 如果采样的轨迹数超过了指定的数量，则停止采样，对齐AgentMove当前测试采样数量
            if counter >=self.prompt_num:
                break

        print("测试集实际轨迹:数量{} ".format(counter))
        return test_trajectory_ids

    def train_traj_sampling(self, user_ids):
        counter = 0
        train_trajectory_ids = []
        tar_user_ids = []
        user_traj = {}
        for user_id in user_ids:
            traj_ids = self.data[self.data['user_id'] == user_id]['DL_traj_id'].values

            # 按照轨迹数量筛选测试用户
            if self.dataset_name in ["Shanghai_Weibo", "Shanghai_ISP"]:
                if len(traj_ids)==0:
                    continue
                else:
                    user_traj[user_id] = []
            else:
                if len(set(self.DL_train_trajectory_ids)&set(traj_ids)) < 12 or len(set(self.DL_train_trajectory_ids)&set(traj_ids)) > 50: # 每个用户最多轨迹数限制为50;最少的限制是考虑到测试集
                    continue
                else:
                    user_traj[user_id] = []

            traj_count=0  # 每个用户取前7个轨迹
            sorted_train_traj = sorted(list(set(self.DL_train_trajectory_ids)&set(traj_ids)))
            for traj_id in sorted_train_traj:
                user_traj[user_id].append(traj_id)
                traj_count += 1
                if traj_count>7:
                    break
        all_users = sorted(user_traj, key=lambda x: len(user_traj[x]), reverse=True)  #按轨迹数从多到少排序
        for user_id in all_users:
            counter += len(user_traj[user_id])
            tar_user_ids.append(user_id)
            train_trajectory_ids.extend(user_traj[user_id])
            if counter > len(all_users)*7:
                break  
        print("训练集实际轨迹数量:{}，用户数：{}".format(counter, len(tar_user_ids)))      
        return train_trajectory_ids, tar_user_ids

    def val_traj_sampling(self, user_ids):
        counter = 0
        val_trajectory_ids = []
        for user_id in user_ids:
            traj_ids = self.data[self.data['user_id'] == user_id]['DL_traj_id'].values

            # 按照轨迹数量筛选测试用户
            if self.dataset_name in ["Shanghai_Weibo", "Shanghai_ISP"]:
                if len(traj_ids)==0:
                    continue
            else:
                if len(set(self.DL_val_trajectory_ids)&set(traj_ids)) < 3:
                    continue
                if len(set(self.DL_val_trajectory_ids)&set(traj_ids)) > self.traj_max_len:
                    continue

            traj_count=0
            sorted_val_traj = sorted(list(set(self.DL_val_trajectory_ids)&set(traj_ids)))
            for traj_id in sorted_val_traj:
                val_trajectory_ids.append(traj_id)
                counter += 1
                traj_count += 1
                if self.sample_one_traj_of_user:
                    break
                else:
                    if traj_count>self.max_sample_trajectories:
                        break
        print("验证集集实际轨迹数量:{} ".format(counter))      
        return val_trajectory_ids

    def get_traj_in_test_user(self, traj_user_map, mode):
        tar_user_ids = set([traj_user_map[traj] for traj in self.DL_test_trajectory_ids])
        tar_traj_ids = []
        if mode == 'train':
            for traj in self.DL_train_trajectory_ids:
                if traj_user_map[traj] in tar_user_ids:
                    tar_traj_ids.append(traj)
        elif mode == 'validation':
            for traj in self.DL_val_trajectory_ids:
                if traj_user_map[traj] in tar_user_ids:
                    tar_traj_ids.append(traj)
        return tar_traj_ids         


    def get_trajectories(self):
        # order the data by user_id and datetime
        self.data = self.data.sort_values(by=['user_id', 'datetime'])

        if self.trajectory_mode == 'user_split':
            # get the unique user ids
            unique_user_ids = self.data['user_id'].unique()
            train_users, test_users = train_test_split(unique_user_ids,
                                                       test_size=self.test_size,
                                                       random_state=self.seed)
            train_data = self.data[self.data['user_id'].isin(train_users)]
            test_data = self.data[self.data['user_id'].isin(test_users)]

            # for each user in test data, we retrieve the historical stays (iloc[-(M+N):-N]) and the context stays (iloc[-N:])
            print('Processing training and test split using method user_split...')
            for user_id in tqdm.tqdm(test_data['user_id'].unique()):
                user_data = test_data[test_data['user_id'] == user_id]
                historical_data = user_data.iloc[-(self.history_stays + self.context_stays):-self.context_stays][['hour', 'weekday', 'venue_category_name', 'venue_id', 'address']].values
                historical_pos = user_data.iloc[-(self.history_stays + self.context_stays):-self.context_stays][['longitude','latitude']].values
                historical_addr = user_data.iloc[-(self.history_stays + self.context_stays):-self.context_stays]['address'].values
                context_data = user_data.iloc[-self.context_stays:][['hour', 'weekday', 'venue_category_name', 'venue_id','address']].values
                context_pos = user_data.iloc[-self.context_stays:][['longitude','latitude']].values
                context_addr = user_data.iloc[-self.context_stays:]['address'].values
                ground_truth = context_data[-1][-1]
                ground_pos = context_pos[-1]
                ground_addr = context_addr[-1]
                target_data = context_data[-1][:-1].tolist()
                target_pos = context_pos[-1].tolist()
                target_addr = context_addr[-1].tolist()
                target_data.append('<next_place_id>')
                context_data = context_data[:-1]
                context_pos = context_pos[:-1]
                context_addr = context_addr[:-1]
    

                user_data_dict = {
                    'historical_stays': historical_data.tolist(),
                    'context_stays': context_data.tolist(),
                    'historical_pos': historical_pos.tolist(),
                    'historical_addr': historical_addr.tolist(),
                    'context_pos': context_pos.tolist(),
                    'context_addr': context_addr.tolist(),
                    'target_stay': target_data,
                    'target_pos': target_pos,
                    'target_addr': target_addr,
                }
                ground_data = {
                    'ground_stay':ground_truth,
                    'ground_pos':ground_pos,
                    'ground_addr':ground_addr,
                }

                self.test_dictionary[str(user_id)] = user_data_dict
                self.true_locations[str(user_id)] = ground_data

        elif self.trajectory_mode == 'trajectory_split':

            def _group_by_time_window(data):
                # Sort by timestamp
                data = data.sort_values('datetime')
                # Start time for the first group
                start_time = data.iloc[0]['datetime']
                # Initialize trajectory ID

                trajectory_id = self.data['traj_id'].max()
                time_window = pd.Timedelta(hours=self.hour_bins)
                for i in range(1, len(data)):
                    # If the current timestamp is outside the 72-hour window, increment the trajectory ID
                    if data.iloc[i]['datetime'] > start_time + time_window:
                        trajectory_id += 1
                        start_time = data.iloc[i]['datetime']
                    data.iloc[i, data.columns.get_loc('traj_id')] = trajectory_id

                return data

            if self.dataset_name in ["Shanghai"]:
                pass
            else:
                # Applying the function to group by ID and then by time window
                self.data['traj_id'] = 0
                self.data = self.data.groupby('user_id').apply(_group_by_time_window).reset_index(drop=True)
            

            ################## add processing for running DL baselines
            unique_id = 0
            user_groups = self.data.groupby('user_id')['traj_id'].unique().to_dict()
            traj_id_map = {}
            traj_user_map = {}
            for user, traj_ids in user_groups.items():
                for traj_id in traj_ids:
                    traj_id_map[(user, traj_id)] = unique_id
                    unique_id += 1
            # Mapping original traj_ids to globally unique DL_traj_id
            self.data['DL_traj_id'] = self.data.apply(lambda row: traj_id_map[(row['user_id'], row['traj_id'])], axis=1)
            for (user, traj_id), unique_traj in traj_id_map.items():
                if unique_traj not in traj_user_map:
                    traj_user_map[unique_traj] = user
                else:
                    print("Not unique encode for trajories!")
            ###################


            address_cols =  ["admin", "subdistrict", "poi", "street"]
            self.data[address_cols] = self.data[address_cols].replace(['NaN', 'NAN', 'nan', None, np.nan, '', 'None', 'none', 'NULL', 'N/A', 'n/a', 'missing', ' ', -999, 999999], '')
            # For each user in FSQ dataset, we retrieve the list of trajectory ids:
            # - the first 70% of the trajectory ids are used for training
            # - the next 10% of the trajectory ids are used for validation
            # - the last 20% of the trajectory ids are used for testing

            # For each user in WWW2019 dataset
            # - the first 40% of the trajectory ids are used for training
            # - the next 10% of the trajectory ids are used for validation
            # - the last 50% of the trajectory ids are used for testing
            print('Processing training and test split using method trajectory_split...')

            DL_train_trajectory_ids = []
            DL_val_trajectory_ids = []
            DL_test_trajectory_ids = []
            for user_id in tqdm.tqdm(self.data['user_id'].unique()):
                # put the user in the dictionaries
                self.test_dictionary[str(user_id)] = {}
                self.true_locations[str(user_id)] = {}
                self.align_dictionary[str(user_id)] = {}

                # exclude users with less than traj_min_len trajectories id
                if len(self.data[self.data['user_id'] == user_id]['traj_id'].unique()) < self.traj_min_len:
                    continue

                trajectory_ids = self.data[self.data['user_id'] == user_id]['traj_id'].unique()
                if self.dataset_name in ["Shanghai", "Shanghai_Weibo"]:# WWW2019 Shanghai-ISP
                    train_trajectory_ids = trajectory_ids[:int(0.5 * len(trajectory_ids))]
                    test_trajectory_ids = trajectory_ids[int(0.5 * len(trajectory_ids)):]
                    train_ids = trajectory_ids[:int(0.4 * len(trajectory_ids))].tolist()
                    train_sample_ids = random.sample(train_ids, int(self.train_sample*len(train_ids)/0.7))
                    val_ids = trajectory_ids[int(0.4 * len(trajectory_ids)):int(0.5 * len(trajectory_ids))]
                else:
                    train_trajectory_ids = trajectory_ids[:int(0.8 * len(trajectory_ids))]
                    train_ids = trajectory_ids[:int(0.7 * len(trajectory_ids))].tolist()
                    train_sample_ids = random.sample(train_ids, int(self.train_sample*len(train_ids)/0.7))
                    val_ids = trajectory_ids[int(0.7 * len(trajectory_ids)):int(0.8 * len(trajectory_ids))]
                    test_trajectory_ids = trajectory_ids[int(0.8 * len(trajectory_ids)):]

                # for each trajectory id in test trajectory ids, we retrieve the historical stays
                # (location ids and time from training) and the context stays (location ids and time
                # from testing with specific trajectory id)

                test_ids = []
                for i, trajectory_id in enumerate(test_trajectory_ids):
                    # exclude this trajectory if it has less than 4 stays
                    if len(self.data[(self.data['user_id'] == user_id) & (self.data['traj_id'] == trajectory_id)]) < 4:
                        continue
                    
                    test_ids.append(trajectory_id)
                    if self.base_name == 'AgentMove': # default data processing for LLM based methods, all llm methods use this
                        venue_id_type = "venue_id_int" if self.use_int_venue else "venue_id" 
                        cared_column_list = ['hour', 'weekday', 'venue_category_name', venue_id_type, "admin", "subdistrict", "poi", "street"]
                        addres_column_list = ["admin", "subdistrict", "poi", "street"]
                        coor_column_list = ['longitude', 'latitude']

                        historical_data = self.data[(self.data['user_id'] == user_id) & (self.data['traj_id'].isin(train_trajectory_ids))][cared_column_list].values
                        historical_addr = self.data[(self.data['user_id'] == user_id) & (self.data['traj_id'].isin(train_trajectory_ids))][addres_column_list].values
                        historical_pos  = self.data[(self.data['user_id'] == user_id) & (self.data['traj_id'].isin(train_trajectory_ids))][coor_column_list].values

                        context_data = self.data[(self.data['user_id'] == user_id) & (self.data['traj_id'] == trajectory_id)][cared_column_list].values
                        context_addr = self.data[(self.data['user_id'] == user_id) & (self.data['traj_id'] == trajectory_id)][addres_column_list].values
                        context_pos  = self.data[(self.data['user_id'] == user_id) & (self.data['traj_id'] == trajectory_id)][coor_column_list].values

                        ground_truth = context_data[-1][3]
                        ground_pos = context_pos[-1]
                        ground_addr = context_addr[-1]

                        target_data = context_data[-1][:2].tolist()
                        target_data.append('<next_place_id>')
                        target_data.append('<next_place_address>')
                        
                        context_data = context_data[:-1]
                        context_pos = context_pos[:-1]
                        context_addr = context_addr[:-1]
                        
                        user_data_dict = {
                            'historical_stays': historical_data.tolist()[-self.history_stays:],
                            'historical_pos': historical_pos.tolist()[-self.history_stays:],
                            'historical_addr': historical_addr.tolist()[-self.history_stays:],
                            'historical_stays_long': historical_data.tolist(),
                            'historical_addr_long': historical_addr.tolist(),
                            'context_stays': context_data.tolist(),
                            'context_pos': context_pos.tolist(),
                            'context_addr': context_addr.tolist(),
                            'target_stay': target_data,
                        }
                        ground_data = {
                            'ground_stay':ground_truth,
                            'ground_pos':ground_pos.tolist(),
                            'ground_addr':ground_addr.tolist(),
                        }

                        align_columns = ["city", "user_id", "venue_id", "utc_time","longitude", "latitude", "venue_category_name", "venue_id_int"]
                        historical_data_align = self.data[(self.data['user_id'] == user_id) & (self.data['traj_id'].isin(train_trajectory_ids))][align_columns].values
                        context_data_align = self.data[(self.data['user_id'] == user_id) & (self.data['traj_id'] == trajectory_id)][align_columns].values
                        alignment_data = {
                            "historical_stays_long": historical_data_align.tolist(),
                            "context_stays": context_data_align.tolist()
                        }
                        self.test_dictionary[str(user_id)][str(trajectory_id)] = user_data_dict
                        self.true_locations[str(user_id)][str(trajectory_id)] = ground_data
                        self.align_dictionary[str(user_id)][str(trajectory_id)] = alignment_data
                
                # Map local traj_ids to DL_traj_id for train, val, and test
                DL_train_trajectory_ids.extend([traj_id_map[(user_id, tid)] for tid in train_sample_ids])
                DL_val_trajectory_ids.extend([traj_id_map[(user_id, tid)] for tid in val_ids])
                DL_test_trajectory_ids.extend([traj_id_map[(user_id, tid)] for tid in test_ids])
                    # Store results
            self.DL_train_trajectory_ids = DL_train_trajectory_ids
            self.DL_val_trajectory_ids = DL_val_trajectory_ids
            self.DL_test_trajectory_ids = DL_test_trajectory_ids

            user_ids = self.data['user_id'].unique()
            # 训练集过滤获得有效用户，仅在有效用户中进行选择
            # self.DL_train_trajectory_ids, tar_user_ids = self.train_traj_sampling(user_ids)
            # self.DL_val_trajectory_ids = self.val_traj_sampling(tar_user_ids)
            self.DL_test_trajectory_ids = self.test_traj_sampling(user_ids)
            if self.dataset_name not in ["Shanghai_ISP"]:
                # 使用出现在测试集的用户的轨迹.对于Shanghai_ISP不做此处理
                DL_train_trajectory_ids = self.get_traj_in_test_user(traj_user_map, 'train')
                DL_val_trajectory_ids = self.get_traj_in_test_user(traj_user_map, 'validation')
            # user_ids, DL_train_trajectory_ids, DL_val_trajectory_ids, test_trajectory_ids
            self.get_baseline(user_ids, DL_train_trajectory_ids, DL_val_trajectory_ids, self.DL_test_trajectory_ids)


        # save the test dictionary and the true locations dictionary
        extra_file_name = ""
        if self.use_int_venue:
            extra_file_name = "_int"
        utils.create_dir(self.save_dir)
        with open(os.path.join(self.save_dir, 'test_dictionary_'+self.dataset_name+'_'+self.trajectory_mode+extra_file_name+'.json'), 'w') as f:
            json.dump(self.test_dictionary, f)
        with open(os.path.join(self.save_dir, 'true_locations_'+self.dataset_name+'_'+self.trajectory_mode+extra_file_name+'.json'), 'w') as f:
            json.dump(self.true_locations, f)
        with open(os.path.join(self.save_dir, 'align_locations_'+self.dataset_name+'_'+self.trajectory_mode+extra_file_name+'.json'), 'w') as f:
            json.dump(self.align_dictionary, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city_name', type=str, default="Shanghai")
    parser.add_argument('--trajectory_mode', type=str, default="trajectory_split", choices=["trajectory_split"])
    parser.add_argument("--historical_stays", type=int, default=15)
    parser.add_argument('--context_stays', type=int, default=6)
    parser.add_argument('--max_sample_trajectories', type=int, default=100, help="对小城市需要采样多条轨迹来测试模型以获得更加置信的结果,优先级低于sample_one")
    parser.add_argument('--use_int_venue', action='store_true', help='Use int Venue ID')
    parser.add_argument('--base_name', type=str, choices=['AgentMove','STHM','GETNext'])  #If you want to run the main Agentmove experiment,please set the param to 'AgentMove'.)
    parser.add_argument('--train_sample', type=float, default=0.7, choices=[0.7,0.5,0.3,0.1])

    args = parser.parse_args()
    random.seed(1234)
    dataset = Dataset(
        base_name=args.base_name,
        dataset_name=args.city_name,
        traj_min_len=2 if args.city_name in ["Shanghai", "Shanghai_Weibo"] else 3, # WWW2019-Shanghai-ISP
        trajectory_mode=args.trajectory_mode, 
        historical_stays=args.historical_stays,
        context_stays=args.context_stays,
        save_dir=PROCESSED_DIR,
        use_int_venue=args.use_int_venue,
        train_sample = args.train_sample
        )