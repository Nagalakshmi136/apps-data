import pandas as pd
from fetchers.app_store import scrape_all_apps

# from tqdm import tqdm
import time
import asyncio

# df = pd.read_csv("app_ids/app_store.csv")
# app_ids = df["app_id"]
# app_titles = df["app_name"]
# for i in tqdm(range(len(app_ids))):
#     print(f'{app_titles[i]} started fetching data')
#     fetch_and_store_app_info(app_ids[i], app_titles[i])
#     print(f'{app_titles[i]} fetched successfully')


df = pd.read_csv("app_ids/app_store.csv")
app_list = list(zip(df["app_name"], df["app_id"]))
start_time = time.time()
asyncio.run(scrape_all_apps(app_list))
end_time = time.time()

print(f"Total time taken: {end_time - start_time:.2f} seconds")
