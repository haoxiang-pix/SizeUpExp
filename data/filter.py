import json
import os
import shutil

output_dir = './filtered_frontal_fullbody_regular/'
os.makedirs(output_dir, exist_ok=True)

vlm_out = json.load(open('/data/lihaox/crawl_raw_data_regular/vlm.json'))
for fpath, res in vlm_out.items():
    if res == 'YES':
        shutil.copy(fpath, f'{output_dir}/{fpath.split("/")[-1]}')
