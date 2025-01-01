import os
import pandas as pd
import json
import time
import httpx
from openai import OpenAI
from tqdm import tqdm
import json_repair
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

import threading
from queue import Queue, Empty


from config import DATASET, NOMINATIM_PATH, NO_ADDRESS_TRAJ_DIR, CITY_DATA_DIR, ADDRESS_L4_DIR, ADDRESS_L4_FORMAT_MODEL, PROXY, EXP_CITIES


PARALLEL_WORKERS = 50
EXP_CITIES = ["Shanghai_Weibo", "Shanghai_ISP"]


def get_normalize_city_name(city_name):
    city_mapping = {"Cape Town":"CapeTown", "New York": "NewYork", "San Francisco": "SanFrancisco", "Sao Paulo": "SaoPaulo"}
    return city_mapping.get(city_name, city_name)

def get_response(address):
    system_messages = [{"role": "system", "content": "You are a helpful assistant for Address Parsing."}]
    client = OpenAI(
            api_key=os.environ["OpenAI_API_KEY"],
            http_client=httpx.Client(proxies=PROXY),
        )
    prompt_text = address + """Please get the Administrative Area Name, subdistrict name/neighbourhood name,access road or feeder road name, building name/POI name.
    Present your answer in a JSON object with:'administrative' (the Administrative Area Name) ,'subdistrict' (subdistrict name/neighbourhood name),'poi'(building name/POI name),'street'(access road or feeder road name which POI/building is on).
                Do not include the key if information is not given.Do not output other content."""
    try:
        full_text = single_chat(client, system_messages, prompt_text)
    except:
        raise Exception
    return full_text


@retry(wait=wait_random_exponential(min=3, max=60), stop=stop_after_attempt(5))
def single_chat(client, system_messages, prompt_text):
    response = client.chat.completions.create(
                model=ADDRESS_L4_FORMAT_MODEL,
                messages=system_messages + [{"role": "user", "content": prompt_text}],
                max_tokens=200,
                temperature=0
            )
    full_text = response.choices[0].message.content
    return full_text

def process_address(city, venue, address, venue_category_name=None):
    try:
        res_addr = get_response(address)
        res_dict = json_repair.repair_json(res_addr, return_objects=True)
        if DATASET == "gowalla":
            res_dict["venue_category_name"] = venue_category_name

        return (city, venue, res_dict, None)
    except json.JSONDecodeError as e:
        return (city, venue, None, f"JSONDecodeError for address {address}: {e}")
    except Exception as e:
        return (city, venue, None, f"Error processing address {address}: {e}")

class Saver:
    def __init__(self, output_path, append=True):
        self.running = False
        self.output_path = output_path
        self.q = Queue()
        self.thread = None
        self.append = append

    def execute(self):
        with open(self.output_path, 'a' if self.append else 'w') as f:
            while self.running or not self.q.empty():
                try:
                    item = self.q.get(block=True, timeout=10)
                    f.write(json.dumps(item))
                    f.write("\n")
                    self.q.task_done()
                except Empty:
                    continue

    def write_item(self, item: dict):
        if self.running:
            self.q.put(item)
        else:
            raise RuntimeError("The Saver needs to be started before writing.")

    def run(self):
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self.execute, daemon=True)
        self.thread.start()

    def stop(self):
        if not self.running:
            raise RuntimeError("Saver not started.")

        self.running = False
        self.thread.join()
        self.thread = None

    def __enter__(self):
        self.run()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False


if __name__ == "__main__":
    for addr_file in os.listdir(NOMINATIM_PATH):
        done_cities = []
        for pos_file in os.listdir(CITY_DATA_DIR):
            city = pos_file.split("_")[0]
            done_cities.append(city)    
        city = addr_file.split(".")[0]
        if city in done_cities:
            continue
        
        if city not in EXP_CITIES:
            continue

        print(f"Start resolving the address of {city}... ")
        file_path = os.path.join(ADDRESS_L4_DIR, f'./{city}_addr_dict.json')
        if not os.path.exists(file_path):
            city_addr_dict = {}  
            addr_data = pd.read_csv(os.path.join(NOMINATIM_PATH,addr_file), sep="\t")
            with Saver(os.path.join(ADDRESS_L4_DIR, f'./{city}_addr.txt')) as s:
                with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
                    futures = []
                    for _, row in tqdm(addr_data.iterrows(),total=addr_data.shape[0]):
                        venue = row['venue_id']
                        address = row['address']
                        if DATASET == "TIST2015":
                            futures.append(executor.submit(process_address, city, venue, address))
                        elif DATASET == "gowalla":
                            venue_category_name = row['venue_category_name']
                            futures.append(executor.submit(process_address, city, venue, address, venue_category_name))      

                    for future in tqdm(as_completed(futures)):
                        city, venue, res_dict, error = future.result()
                        key = f"{city}_{venue}"
                        if error:
                            print(error)
                        else:
                            if key not in city_addr_dict:
                                s.write_item({key: res_dict})
                                city_addr_dict[key] = res_dict
                        
            with open(file_path, 'w', encoding="utf-8") as file:
                json.dump(city_addr_dict, file, ensure_ascii=False)
        else:
            # with open(file_path, 'r', encoding="utf-8") as file:
            #     city_addr_dict = json.load(file, ensure_ascii=False)
            city_addr_dict = {}
            print(f"Start downloading the address of {city}... ")
            with open(os.path.join(ADDRESS_L4_DIR, f'./{city}_addr.txt'), 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    try:
                        json_obj = json.loads(line)
                        city_addr_dict.update(json_obj)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
            print(city_addr_dict[key] for key in list(city_addr_dict.keys())[:10])
        
        print(f"Start matching address of {city}....")
        city_data = pd.read_csv(os.path.join(NO_ADDRESS_TRAJ_DIR, f'{city}_filtered.csv'))
        
        city_data["city_normalize"] = city_data["city"].apply(lambda x: get_normalize_city_name(x))
        city_data['key'] = city_data.apply(lambda x: f"{x['city_normalize']}_{x['venue_id']}", axis=1)
        city_data = city_data[city_data['key'].isin(city_addr_dict.keys())]
        
        city_data['admin'] = city_data['key'].apply(lambda key: city_addr_dict[key].get("administrative", ""))
        city_data['subdistrict'] = city_data['key'].apply(lambda key: city_addr_dict[key].get("subdistrict", ""))
        city_data['poi'] = city_data['key'].apply(lambda key: city_addr_dict[key].get("poi", ""))
        city_data['street'] = city_data['key'].apply(lambda key: city_addr_dict[key].get("street", ""))
        if DATASET == "gowalla":
            city_data['venue_category_name'] = city_data['key'].apply(lambda key: city_addr_dict[key].get("venue_category_name", ""))
        
        city_data = city_data.drop(columns=['key'])
        
        city_data.to_csv(os.path.join(CITY_DATA_DIR, f'{city}_filtered.csv'), encoding="utf-8", index=False)
