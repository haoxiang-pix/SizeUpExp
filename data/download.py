import tqdm
import random
import cv2
import requests
import os, sys, argparse
import json
import urllib.parse

base_url = 'https://pixdatanest.blob.core.windows.net/'
sas = '?sv=2023-11-03&st=2024-07-30T18%3A39%3A23Z&se=2024-07-31T18%3A39%3A23Z&sr=c&sp=rl&sig=b8W6XmCcFIG6CEmfHe1NB%2F7g9UFCXIF65JKYy9oP7SA%3D'
output_dir = '/data/lihaox/crawl_raw_data_regular/'
os.makedirs(output_dir, exist_ok=True)

data = json.load(open('sample_by_source.json', 'r'))
selected_sources = data.keys() - set(['www.buycurvy.com','www.bloomchic.com','www.bfreeaustralia.com'])
num_per_source = 1000

all_urls = []
for source, urls in data.items():
    if source not in selected_sources: continue
    subdir = os.path.join(output_dir, source)
    os.makedirs(subdir, exist_ok=True)
    random.shuffle(urls)
    for url in tqdm.tqdm(urls[:num_per_source]):
        if type(url) != str:
            continue
        filename = url.split('/')[-1]
        fpath = os.path.join(subdir, filename)
        if os.path.exists(fpath):
            try:
                I = cv2.imread(fpath)
                continue
            except:
                pass
        download_url = f'{base_url}{urllib.parse.quote(url)}{sas}'
        all_urls.append([download_url, fpath])

import multiprocessing
def download(params):
    download_url, fpath = params
    try:
        with open(fpath, 'wb') as f:
            res = requests.get(download_url)
            f.write(res.content)
        return None
    except:
        return [download_url, fpath]

failed = []
for ret in tqdm.tqdm(multiprocessing.Pool(24).imap_unordered(download, all_urls), total=len(all_urls)):
    if ret is not None:
        failed.append(ret)
print (len(failed))
print (failed)
