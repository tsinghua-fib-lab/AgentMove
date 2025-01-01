import os
import re
import glob
import json
import jsmin
import argparse
import json_repair
import numpy as np

from config import EXP_CITIES, PROCESSED_DIR
from token_count import TokenCount

def create_dir(dir):
    # if dir does not exist, create it
    if not os.path.exists(dir):
        os.makedirs(dir)


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


def match_prediction(text):
    match = re.search(r'[Pp]rediction(.*?)[Rr]eason', text, re.DOTALL)
    if match:
        prediction_text = match.group(1)
        place_ids = re.findall(r'\b[0-9a-f]{24}\b', prediction_text)
    else:
        place_ids = []
    return place_ids


def token_count(text):
    tc = TokenCount(model_name="gpt-3.5-turbo")
    return tc.num_tokens_from_string(text)

def extract_json(full_text):
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
            prediction = output_json.get('prediction')
            if len(prediction)==0:
                prediction = match_prediction(output_json)
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
                prediction = match_prediction(full_text)
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