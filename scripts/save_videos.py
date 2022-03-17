import cv2
import numpy as np
from pathlib import Path
import os
import time
from lqcv.data import data_reader


localtime = time.asctime(time.localtime(time.time()))
now = time.strftime("%Y.%m.%d %H:%M:%S", time.localtime())
day, hms = now.split(' ')
