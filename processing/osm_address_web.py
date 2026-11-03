import os
import threading
from queue import Queue, Empty

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

from config import NO_ADDRESS_TRAJ_DIR, NOMINATIM_PATH, EXP_CITIES

CURRENT_CITY = EXP_CITIES[0]  # "Shanghai", taken from config.py
OUT_FILE = os.path.join(NOMINATIM_PATH, f"{CURRENT_CITY}_address.txt")

# Create the geocoder ONCE so the 1-request-per-second limit actually applies.
# OpenStreetMap asks for an identifying user agent: put your own email here.
geolocator = Nominatim(user_agent="agentmove-thesis-siit (peldenchewang2003@gmail.com)", timeout=10)
geocode = RateLimiter(geolocator.reverse, min_delay_seconds=1.1,
                      max_retries=3, error_wait_seconds=5, swallow_exceptions=True)


def reverse_geocode(lat, lon):
    location = geocode((lat, lon), exactly_one=True, language='en')
    return location.address if location else None


def load_done():
    """Venue IDs already saved, so a rerun can resume where it stopped."""
    done = set()
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    done.add(parts[1])
    return done


def process_map(city, venue_city, done):
    total = len(venue_city)
    for i, (venue, (lon, lat)) in enumerate(venue_city.items(), 1):
        if str(venue) in done:
            continue
        address = reverse_geocode(lat, lon)
        if address:
            q.put({"city": city, "venue_id": venue, "lng": lon, "lat": lat, "address": address})
        if i % 50 == 0:
            print(f"{i}/{total} venues looked up")


def save_worker():
    with open(OUT_FILE, 'a') as f:
        while running or not q.empty():
            try:
                item = q.get(block=True, timeout=10)
                f.write(f'{item["city"]}\t{item["venue_id"]}\t{item["lng"]}\t{item["lat"]}\t{item["address"]}\n')
                f.flush()
                q.task_done()
            except Empty:
                continue


if __name__ == '__main__':
    os.makedirs(NOMINATIM_PATH, exist_ok=True)
    q = Queue()
    running = True

    thread = threading.Thread(target=save_worker, daemon=True)
    thread.start()

    venue_map = {}
    print("reading city files...")
    for file in os.listdir(NO_ADDRESS_TRAJ_DIR):
        if file != f"{CURRENT_CITY}_filtered.csv":   # only the main dataset for this city
            continue
        print(f"reading {file}...")
        fs = pd.read_csv(os.path.join(NO_ADDRESS_TRAJ_DIR, file))
        lon_col = "longitude" if "longitude" in fs.columns else "lon"
        lat_col = "latitude" if "latitude" in fs.columns else "lat"
        city_venues = venue_map.setdefault(CURRENT_CITY, {})
        for _, row in fs.iterrows():
            venue = row['venue_id']
            if venue not in city_venues:
                city_venues[venue] = (row[lon_col], row[lat_col])

    done = load_done()
    for city, items in venue_map.items():
        print(f"{city}: {len(items)} unique venues, {len(done)} already saved")
        process_map(city, items, done)

    q.join()          # wait until every address has been written
    running = False
    thread.join()
    print("finished")