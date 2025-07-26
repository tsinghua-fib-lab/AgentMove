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
from config import DATASET, NOMINATIM_PATH, NO_ADDRESS_TRAJ_DIR, NOMINATIM_DEPLOY_SERVER, NOMINATIM_DEPLOY_WORKERS, EXP_CITIES


# you can deploy nominatim service by referring to https://github.com/mediagis/nominatim-docker/tree/master/4.4
###########new version##################
@retry(wait=wait_random_exponential(min=3, max=60), stop=stop_after_attempt(10))
def reverse_geocode_v2(city, venue, lon, lat):
    # https://nominatim.org/release-docs/develop/api/Reverse/
    url = "http://{}/reverse?format=jsonv2&lat={}&lon={}&zoom=18&addressdetails=1&accept-language=en-US".format(NOMINATIM_DEPLOY_SERVER, lat, lon)
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
    item_info = {
                "city": city,
                "venue_id": venue,
                "lng": lon,
                "lat": lat,
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

    with multiprocessing.Pool(NOMINATIM_DEPLOY_WORKERS) as pool:
        results = pool.starmap(geocode_extract, coor_list)
    for res in results:
        city_res.append(res)
    
    data = pd.json_normalize(city_res)
    os.makedirs(NOMINATIM_PATH, exist_ok=True)
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

        if city not in EXP_CITIES:
            continue

        cities.append(city)
        venue_map[city] = {}
        for _,row in fs.iterrows():
            venue = row['venue_id']
            lon = row['longitude']
            lat = row['latitude']
            if venue not in venue_map[city]:
                venue_map[city][venue] = (lon, lat)
        
    for city,items in venue_map.items():
        print(city,len(items))
    for city in cities:
        process_map_v2(city, venue_map[city])
