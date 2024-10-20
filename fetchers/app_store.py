import asyncio
import aiohttp
import aiofiles
from aiohttp import ClientSession
import pandas as pd
from itunes_app_scraper.scraper import AppStoreScraper
from app_store_scraper import AppStore
from pathlib import Path
import random
from datetime import datetime
import json
import time
from utils.common_utils import serialize_datetime

async def fetch_app_details(app_id: str):
    scraper = AppStoreScraper()
    return scraper.get_app_details(app_id)

async def fetch_reviews(app_name: str, app_id: str, after: datetime):
    my_app = AppStore(country="us", app_name=app_name, app_id=app_id)
    all_reviews = []

    await asyncio.sleep(random.uniform(1, 3))  # Random delay between requests
    my_app.review(after=after)

    print(f"{app_name}: Fetched {len(my_app.reviews)} reviews. Total: {len(all_reviews)}")
    return my_app.reviews

async def fetch_and_store_app_info(session: ClientSession, app_id: str, app_name: str):
    app_name = app_name.replace(" ", "")
    app_data_file = f"data/appstore/{app_name}.json"

    if Path(app_data_file).exists():
        print(f"{app_name} data already exists. Skipping.")
        return

    try:
        # Fetching app information
        app_info = await fetch_app_details(app_id)

        # Fetching reviews from app launch date
        release_date = app_info["releaseDate"]
        release_date_format = "%Y-%m-%dT%H:%M:%SZ"
        after = datetime.strptime(release_date, release_date_format)

        all_reviews = await fetch_reviews(app_name, app_id, after)
        app_info["reviews"] = all_reviews
        print(f"{app_name} storing data. Total reviews: {len(all_reviews)}")

        # Incremental saving of fetched data
        app_info_json = json.dumps(app_info, indent=4, default=serialize_datetime)
        async with aiofiles.open(app_data_file, mode="w") as file_writer:
            await file_writer.write(app_info_json)

    except aiohttp.ClientError as e:
        print(f"Network error occurred for {app_name}: {e}. Data not fully saved.")
        # Optionally, save whatever is fetched so far
        async with aiofiles.open(app_data_file, mode="w") as file_writer:
            await file_writer.write(json.dumps(app_info, indent=4, default=serialize_datetime))
    except Exception as e:
        print(f"An error occurred for {app_name}: {e}")
        # Save partially fetched data in case of an error
        async with aiofiles.open(app_data_file, mode="w") as file_writer:
            await file_writer.write(json.dumps(app_info, indent=4, default=serialize_datetime))

async def scrape_all_apps(app_list):
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_and_store_app_info(session, app_id, app_name)
            for app_name, app_id in app_list
        ]
        await asyncio.gather(*tasks)

