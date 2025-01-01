import json
import pandas as pd
import tqdm
import json
import torch
import os

DEVICE = 'cpu'
from config import DATASET, TIST2015_DATA_DIR, GOWALLA_DATA_DIR, NO_ADDRESS_TRAJ_DIR, EXP_CITIES

def haversine_torch(lonlat1, lonlat2):
    """
    Calculate the distance from the longitude and latitude vector longlat1 to longlat2. 
    Return a distance matrix Mat_MxN, where M is the length of longlat1 and N is the length of longlat2.
    """
    lon1, lat1 = lonlat1[:, 0], lonlat1[:, 1]
    lon2, lat2 = lonlat2[:, 0], lonlat2[:, 1]
    lon1, lat1, lon2, lat2 = map(torch.deg2rad, [lon1, lat1, lon2, lat2])
    lon1 = lon1[:, None]
    lat1 = lat1[:, None]
    lon2 = lon2[None, :]
    lat2 = lat2[None, :]

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.asin(torch.sqrt(a))
    r = 6371
    return c * r


if __name__ == '__main__':
    input_path = TIST2015_DATA_DIR
    output_path = NO_ADDRESS_TRAJ_DIR
    cities_file = "dataset_TIST2015_Cities.txt"
    exp_cities = EXP_CITIES
 
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok = True)
    print("read original global fourquare data...")
    cities_info = []
    with open(os.path.join(input_path, cities_file), 'r', encoding='utf8') as f:
        cities = [city for city in f.readlines() if city.strip() != '']
        for city in tqdm.tqdm(cities):
            city_name, city_lat, city_lng, _, _, _ = city.split('\t')
            cities_info.append({
                'name': city_name,
                'lat': float(city_lat),
                'lng': float(city_lng)
            })

    city_lnglat = torch.tensor([[city['lng'], city['lat']] for city in cities_info]).to(device=DEVICE)

    if DATASET == 'TIST2015':
        checkins_file = "dataset_TIST2015_Checkins.txt"
        pois_file = "dataset_TIST2015_POIs.txt"
        
        print("read poi...")
        pois = []
        with open(os.path.join(input_path,pois_file), 'r', encoding='utf8') as f:
            pois_ = [poi for poi in f.readlines() if poi.strip() != '']
            for line in tqdm.tqdm(pois_):
                poi_id, poi_lat, poi_lng, poi_category, country_code = line.strip().split('\t')
                pois.append({
                    'id': poi_id,
                    'lat': float(poi_lat),
                    'lng': float(poi_lng),
                    'category': poi_category,
                    'country': country_code
                })

        poi_lnglat = torch.tensor([[poi['lng'], poi['lat']] for poi in pois]).to(device=DEVICE)

        
        print("calculate distance...")
        dist = haversine_torch(poi_lnglat, city_lnglat)
        min_index = torch.argmin(dist, dim=1)

        print("mapping POI to cities...")
        min_index = min_index.cpu()
        poi_to_city = {}
        for index, poi in tqdm.tqdm(enumerate(pois), total=len(pois)):
            poi['city'] = cities_info[min_index[index].item()]['name']
            id_ = poi.pop('id')
            poi_to_city[id_] = poi

        venue_dict = {}   #{Venue ID:City Name}
        for venue,data in poi_to_city.items():
            city = data['city']
            if venue not in venue_dict:
                venue_dict[venue] = city
        city_dict = {}  #{City Name:[Venus IDs]}
        for venue,city in venue_dict.items():
            if city not in city_dict:
                city_dict[city] = []
            city_dict[city].append(venue)
        pois_df = pd.read_csv(os.path.join(input_path,pois_file), sep='\t', header=None, names=[
            "Venue ID", "Latitude", "Longitude", "Venue Category Name", "Country Code"
        ])
        checkins_df = pd.read_csv(os.path.join(input_path,checkins_file), sep='\t', header=None, names=[
            "User ID", "Venue ID", "UTC Time", "Timezone Offset"
        ])
        
        for city_name in tqdm.tqdm(city_dict):
            print("processing {} ...".format(city_name))
            result = []
            venues = set(city_dict[city_name])   
            filtered_pois = pois_df[pois_df["Venue ID"].isin(venues)]
            filtered_checkins = checkins_df[checkins_df["Venue ID"].isin(venues)]

            venue_message = {}
            for idx,row in filtered_pois.iterrows():
                venue = row['Venue ID']
                if venue not in venue_message:
                    venue_message[venue] = {"lon":row['Longitude'],"lat":row['Latitude'],"venue_id":row['Venue ID'],"venue_cat_name":row["Venue Category Name"]}
                    
            user_set = set()
            user_dict = {}
            for idx, row in filtered_checkins.iterrows():
                user = row['User ID']
                time = row['Timezone Offset']
                utc_time = row["UTC Time"]
                venue = row['Venue ID'] 
                lon = venue_message[venue]["lon"]
                lat = venue_message[venue]["lat"]
                if venue not in venue_dict:
                    print(venue)
                    continue
                city = venue_dict[venue]
                result.append({"city":city,"user":user,"time":time,"venue_id":venue,"utc_time":utc_time,"lon":venue_message[venue]["lon"],"lat":venue_message[venue]["lat"],"venue_cat_name":venue_message[venue]["venue_cat_name"]})   

            print("output extracted data...")
            print("Filtered POIs:")
            print(filtered_pois.shape)
            print("Filtered Check-ins:")
            print(filtered_checkins.shape)
            result_df = pd.DataFrame(result)
            print("save data...")
            result_df.to_csv(os.path.join(output_path,"{}_filtered.csv".format(city_name)), index=False)
    elif DATASET == 'gowalla':
        input_file_path = GOWALLA_DATA_DIR
        checkins_file = "gowalla_totalCheckins.txt"
        print("read poi...")
        traj = pd.read_csv(os.path.join(input_file_path,checkins_file), sep='\t', names=['user', 'check_in_time', 'lat', 'lon', 'location_id'])
        checkin_lnglat = torch.tensor(traj[['lon', 'lat']].values).float().to(device=DEVICE)
        print("calculate distance...")
        dist = haversine_torch(checkin_lnglat, city_lnglat)
        min_index = torch.argmin(dist, dim=1)

        print("mapping poi to city...")
        min_index = min_index.cpu()
        traj['city'] = [cities_info[idx]['name'] for idx in min_index.numpy()]
        
        for city_name in exp_cities:
            if city_name in traj['city'].values:
                city_data = traj[traj['city'] == city_name]
                output_filename = os.path.join(output_path, f"{city_name}_filtered.csv")
                print(f"save checkin data of {city_name} to {output_filename}")
                city_data.to_csv(output_filename, index=False)
    else:
        print("Invalid dataset name. Please choose 'gowalla' or 'TIST2015'")

