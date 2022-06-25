import os
import shutil
from tqdm import tqdm
from pathlib import Path
import glob

from .misc import is_str


def is_filepath(x):
    return is_str(x) or isinstance(x, Path)


def create_folder(path="./new"):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder


def flatten_recursive(path="../datasets/coco128"):
    # Flatten a recursive directory by bringing all files to top level
    new_path = Path(path + "_flat")
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + "/**/*.*", recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not os.path.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == "":
        return
    dir_name = os.path.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)
