from google_play_scraper import app, Sort, reviews_all
import json
from utils.common_utils import serialize_datetime 
from pathlib import Path

def fetch_and_store_app_info(app_id: str, app_name: str):
    app_name = app_name.replace(" ", "")
    app_data_file = f"data/playstore/{app_name}.json"
    if Path(app_data_file).exists():
        return
    #fetching app information
    app_info = app(app_id)
    app_reviews = reviews_all(app_id, sort=Sort.NEWEST)
    app_info['reviews'] = app_reviews;
    print(f"{app_name} storing data")
    # Storing app information
    app_info_json = json.dumps(app_info, indent=4, default=serialize_datetime)
    with open(app_data_file, "w") as file_writer:
        file_writer.write(app_info_json)
