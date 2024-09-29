import pandas as pd
from fetchers.play_store import fetch_and_store_app_info
from tqdm import tqdm
df = pd.read_csv("app_ids.csv")
app_ids = df["android_appID"]
app_titles = df["app_name"]
for i in tqdm(range(len(app_ids))):
    print(f'{app_titles[i]} started fetching data')
    fetch_and_store_app_info(app_ids[i], app_titles[i])
    print(f'{app_titles[i]} fetched successfully')
    
