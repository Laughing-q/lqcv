import cv2
import os
from pathlib import Path
from tqdm import tqdm
from lqcv.data import Videos
from .image import resize


def videos2images(video_dir, save_dir, interval=1, count_only=False, max_size=None, tail=""):
    """Clip images from videos.

    Args:
        video_dir (str): a dir includes videos or just a video path.
        save_dir (str): a dir to keep images.
        interval (int): save interval, unit is seconds.
        max_size (optional | int): The max size for the image.
        count_only (bool): Only count how many images will be saved, intead of actually saving them.
        tail (str): Special tail for image name.
    """
    reader = Videos(video_dir)
    total = 0
    if count_only:
        for f in tqdm(reader.files, total=len(reader.files)):
            reader.new_video(f)
            total += reader.frames / (reader.fps * interval if reader.fps <= 100 else 25 * interval)
            reader.cap.release()
        print(f"{reader.nf} videos will generate {int(total)} pics({interval}s/pic)")
        return
    os.makedirs(save_dir, exist_ok=True)
    for im, p, s in reader:
        # set interval
        reader.vid_stride = int((reader.fps if reader.fps <= 100 else 25) * interval)
        video_name = Path(Path(p).name).with_suffix("")
        if im is None:
            continue
        h, w = im.shape[:2]
        s += f" WxH:({w}x{h})"
        if max_size:
            im = resize(im, max_size)
            rh, rw = im.shape[:2]
            s += f" resized WxH:({rw}x{rh})"
        s += f" file:{video_name} current: {total}"
        im_name = f"{video_name}_{str(reader.frame).zfill(6)}_{tail}.jpg"
        cv2.imwrite(os.path.join(save_dir, im_name), im)
        total += 1
        print(s)

if __name__ == "__main__":
    videos2images("/home/laughing/Videos", save_dir="./images", count_only=True, interval=1)
