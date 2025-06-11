
import os

# 其他配置
PROXY = "http://127.0.0.1:10190" # you should prepare your proxy server
# EXP_CITIES = ['Tokyo', 'Nairobi', 'NewYork', 'Sydney', 'CapeTown', 'Paris', 'Beijing', 'Mumbai', 'SanFrancisco', 'London', 'SaoPaulo', 'Moscow']
EXP_CITIES = ["Beijing"] # use it for quickly start
# EXP_CITIES = ["Shanghai"] # for WWW2019

#'gowalla'或'TIST2015'
DATASET = 'TIST2015' # format which used by TIST2015 and WWW2019

# Original Data
DATA_PATH = "data/"
TIST2015_DATA_DIR = "{}dataset_tist2015".format(DATA_PATH)       
TSMC2014_DATA_DIR = "{}dataset_tsmc2015".format(DATA_PATH)       
GOWALLA_DATA_DIR = "{}dataset_gowalla".format(DATA_PATH)         
WWW2019_DATA_DIR = "{}dataset_www2019".format(DATA_PATH)         


# Temp Data, used for location address matching
NOMINATIM_DEPLOY_SERVER = os.environ["nominatim_deploy_server_address"] # IP: PORT e.g., 127.0.0.1:18081
NOMINATIM_DEPLOY_WORKERS = 20 # Number of parallel workers for address matching

NO_ADDRESS_TRAJ_DIR = "data/input_trajectories/"  # Trajectory data without addresses after city division from Foursquare, input data for fsq_address_deploy, output data from process_city_data
NO_ADDRESS_WEIBO_TRAJ_DIR = "{}/input/".format(WWW2019_DATA_DIR)
NOMINATIM_PATH = 'data/nominatim/'                # Path where address data is saved after requesting address service, output data for fsq_address_deploy
ADDRESS_L4_DIR = "data/address_L4/"                # Processed and formatted Nominatim address data into a 4-level address structure
ADDRESS_L4_FORMAT_MODEL = "llama4-17b" # Name of the LLM used for 4-level address formatting
ADDRESS_L4_WORKERS = 50                # Number of parallel workers for address formatting

# Final Data
CITY_DATA_DIR = "data/input_trajectories_clean/"  # Path to trajectory data read by data.py, also the output data path from process_city_data_pos
PROCESSED_DIR = "data/processed/"                 # Path to processed trajectory data output by data.py, also the data path read by agent.py

# Results
SUMMARY_SAVE_DIR = "results/summary/"            # analysis.py

# API访问重试参数
# @retry(wait=wait_random_exponential(min=3, max=60), stop=stop_after_attempt(5))
WAIT_TIME_MIN = 3
WAIT_TIME_MAX = 60
ATTEMPT_COUNTER = 10
VLLM_URL = "xxx" # vllm serving API URL settings
