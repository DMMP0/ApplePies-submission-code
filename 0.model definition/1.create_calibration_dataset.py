import os

import numpy as np
import torch

from os import makedirs
from os.path import normpath, join
from torchvision.io import read_image
from random import shuffle
from torchvision.transforms.v2.functional import resize
from tqdm import tqdm
import pickle
import argparse
import json

# CLI args parser
# ---------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    prog='python3 1.create_calibration_dataset.py',
    description='Script used to create the calibration dataset',
    epilog="""

    Example of usage: 
    python3 1.create_calibration_dataset.py --config "./configs/default_hailo_conversion_config.json" """,

    formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('--config', default="./configs/default_hailo_conversion_config.json", type=str,
                    help="""Config file""")

args = parser.parse_args()
# ---------------------------------------------------------------------------------------------------------------------


with open(args.config, 'r') as f:
    config = json.load(f)

# calibration dataset path
calibration_dataset_og_path:str = config['dataset_path'] # <----- CHANGE THIS IN THE CONFIG FILE
LIMIT_SAMPLES:int = config["limit_samples"]
paths = os.listdir(calibration_dataset_og_path)
shuffle(paths)
paths = paths[:LIMIT_SAMPLES]
actual_paths = [normpath(join(calibration_dataset_og_path, x)) for x in paths]
save_folder = config["save_folder"]
makedirs(save_folder, exist_ok=True)

pass # for debug purposes

# useful function
def torch_to_numpy(x:torch.Tensor) -> np.ndarray:
    return x.contiguous().detach().cpu().numpy()

def read_and_prep(img_path:str, size:tuple[int, int]):
    img_t = read_image(img_path)
    img_t = resize(img_t, size=list(size)).to(dtype=torch.float32)
    # hailo needs W, H, C
    img_t = img_t.permute((1,2,0))
    ris = torch_to_numpy(img_t)
    return ris



target_res = config["target_resolutions"]


for res in target_res:
    print(f"Preparing data for res {res}")
    save_name = f"cal_data_{res[0]}_{res[1]}.npy"
    cal_data = np.zeros((LIMIT_SAMPLES, res[0], res[1], 3))
    for i, path in enumerate(tqdm(actual_paths)):
        npy_image = read_and_prep(path, res)
        cal_data[i, :, :, :] = npy_image

    save_path = normpath(join(save_folder, save_name))
    with open(save_path, 'wb') as f:
        pickle.dump(cal_data, f)
    pass

