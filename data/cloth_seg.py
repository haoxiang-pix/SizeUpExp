from api.platform_utils import get_cloth_segment_mask
import multiprocessing
import glob, tqdm
from PIL import Image
import cv2, os
import multiprocessing

input_dir = './filtered_frontal_fullbody_regular/'
output_dir = './filtered_frontal_fullbody_regular_mask/'
os.makedirs(output_dir, exist_ok=True)

fpaths = sorted(glob.glob(f'{input_dir}/*.*'))
def process(fpath):
    try:
        img = Image.open(fpath).convert("RGB")
        clothmask, clothmask_multicls=get_cloth_segment_mask(img)
        output_fpath = f'{output_dir}/{fpath.split("/")[-1]}'.replace(".jpg", ".png")
        cv2.imwrite(output_fpath, clothmask_multicls)
        return None
    except:
        return fpath

for ret in tqdm.tqdm(multiprocessing.Pool(4).imap_unordered(process, fpaths), total=len(fpaths)):
    if ret is not None:
        print(ret)
