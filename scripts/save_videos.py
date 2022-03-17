import cv2
import numpy as np
from pathlib import Path
import os
import time
from lqcv.data import create_reader


localtime = time.asctime(time.localtime(time.time()))
now = time.strftime("%Y.%m.%d %H:%M:%S", time.localtime())
day, hms = now.split(' ')

source =''
save_dir = ''

reader, webcam = create_reader(source)

if webcam:
    bs = len(reader)
else:
    bs = 1

vid_path, vid_writer = [None] * bs, [None] * bs



for image, path, _ in reader:
    if webcam:  # batch_size >= 1
        for i in image:
            p, frame = path[i], reader.count
            save_path = str(save_dir / p.name)  # im.jpg
            if vid_path[i] != save_path:  # new video
                vid_path[i] = save_path
                fps, w, h = 30, image[i].shape[1], image[i].shape[0]
                save_path += '.mp4'
                vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer[i].write(image[i])
    else:
        p, frame = path, getattr(reader, 'frame', 0)
        reader.save(save_path, image)
