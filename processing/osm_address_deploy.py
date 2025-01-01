import os
import json
import threading
import requests
import multiprocessing
import pandas as pd 
import json_repair
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from config import DATASET, NOMINATIM_PATH, NO_ADDRESS_TRAJ_DIR

CURRENT_CITY = "Moscow"
WORKERS = 20
SERVING_IP = ""
port_mapping = {}
PORT = port_mapping.get(CURRENT_CITY, 18081)

###########new version##################
@retry(wait=wait_random_exponential(min=3, max=60), stop=stop_after_attempt(10))
def reverse_geocode_v2(city, venue, lon, lat):
    # https://nominatim.org/release-docs/develop/api/Reverse/
    url = "http://{}:{}/reverse?format=jsonv2&lat={}&lon={}&zoom=18&addressdetails=1&accept-language=en-US".format(SERVING_IP, PORT, lat, lon)
    response = requests.get(url)
    location = json.loads(response.text)
    try:
        address = json.dumps(location["address"], ensure_ascii=False) if location else None
        category = location.get("category", "") if location else ""
    except:
        address = ""
        category = ""
    return address, category


def geocode_extract(city, venue, lon, lat):
    try:
        addr, cate = reverse_geocode_v2(city, venue, lon, lat)
    except:
        addr = ""
        cate = "" 
    if DATASET == 'TIST2015':
        item_info = {
                    "city": city,
                    "venue_id": venue,
                    "lng": lon,
                    "lat": lat,
                    "address": addr 
                }
    elif DATASET == 'gowalla':
        item_info = {
                    "city": city,
                    "venue_id": venue,
                    "lng": lon,
                    "lat": lat,
                    "venue_category_name": cate,
                    "address": addr
                }
    return item_info


def process_map_v2(city, venue_city):
    venue_address = {}
    coor_list = []
    city_res = []
    for venue, coord in venue_city.items():
        if venue not in venue_address:
            lat, lon = coord[1], coord[0]
            coor_list.append((city, venue, lon, lat))

    with multiprocessing.Pool(WORKERS) as pool:
        results = pool.starmap(geocode_extract, coor_list)
    for res in results:
        city_res.append(res)
    
    data = pd.json_normalize(city_res)
    data.to_csv(os.path.join(NOMINATIM_PATH, "{}.csv".format(city)), sep="\t")


def load_address(file_path):
    data = []
    with open(file_path) as fid:
        for i, line in enumerate(fid):
            if i==0:
                continue
            num, city, venue_id, lng, lat, address_str = line.split("\t")
            address_str = address_str.encode().decode('unicode-escape')
            address_dict = json_repair.repair_json(address_str, return_objects=True)
            data.append([city, venue_id, lng, lat, address_dict])
    return data


if __name__ == '__main__':
    venue_map = {}
    cities = []
    print("reading city files...")
    for file in os.listdir(NO_ADDRESS_TRAJ_DIR):
        fs = pd.read_csv(os.path.join(NO_ADDRESS_TRAJ_DIR, file))
        city = file.split("_")[0]
        print(f"processing city {city}...")

        if city != CURRENT_CITY:
            continue

        cities.append(city)
        venue_map[city] = {}
        for _,row in fs.iterrows():
            venue = row['venue_id']
            lon = row['lon']
            lat = row['lat']
            if venue not in venue_map[city]:
                venue_map[city][venue] = (lon, lat)
        
    for city,items in venue_map.items():
        print(city,len(items))
    for city in cities:
        process_map_v2(city, venue_map[city])
