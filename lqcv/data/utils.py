from pathlib import Path
import numpy as np
import cv2
import os.path as osp
import os

IMG_FORMATS = [
    "bmp",
    "jpg",
    "jpeg",
    "png",
    "tif",
    "tiff",
    "dng",
    "webp",
    "mpo",
]  # acceptable image suffixes

VID_FORMATS = [
    "mov",
    "avi",
    "mp4",
    "mpg",
    "mpeg",
    "m4v",
    "wmv",
    "mkv",
    "vdo",
    "flv",
    "ts",
]  # acceptable video suffixes

