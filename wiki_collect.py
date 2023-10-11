import pandas as pd
import numpy as np
import os
import requests
import time
from tqdm import tqdm
import pickle

for letter in "0":
    file = f"not_seen_idxs_{letter}"#.parquet"
    outdir = "tmp_{}".format(file.split(".")[0])

    # # Load from parquet option
    # df = pd.read_parquet("./wiki_data_clean_v8/wiki_2023_index.parquet")
    # df = df[df.file == file]
    # df = list(df.id.values)

    # Load from pickle array
    df = pickle.load(open(file, 'rb'))


    if os.path.exists(outdir):
        raise ValueError("ALREADY COLLECTED: {}".format(file))
    else:
        os.mkdir(outdir)

    url = 'https://en.wikipedia.org/w/api.php'
    params={
        "format": "json",
        "action": "query",
        "prop": "extracts",
        "explaintext": "",
        "pageids": ""
    }
    headers={
        "User-Agent": "Bot4WikipediaData Contact: miracleyoung0723@gmail.com"
    }
    save_every_n = 1000
    arr = []
    for i, val in enumerate(tqdm(df)):
        params["pageids"] = str(val)
        try:
            r = requests.get(url, params=params, headers=headers, timeout=3.0)
            d = r.json()
            arr.append(d)
        except Exception as e:
            arr.append({})
            print(f"ID: {val} Error: {e}")
            
        if i!=0 and (i%save_every_n==0 or i==len(df)-1):
            extract_arr = []
            for val in arr:
                try:
                    page_id = list(val["query"]["pages"].keys())[0]
                    extract_arr.append(val["query"]["pages"][page_id]["extract"])
                except:
                    extract_arr.append("")
            arr = []
                    
            # Open the file in binary write mode
            with open(outdir + "/" + f"{i-save_every_n}_{i}.pkl", 'wb') as pickle_file:
                # Use pickle.dump() to write the array to the file
                pickle.dump(extract_arr, pickle_file)
        
        # Rest the API a little
        time.sleep(0.01)