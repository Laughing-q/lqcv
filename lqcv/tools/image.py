from lqcv.simi.hash import img_hash, cmpImgHash_np, pHash, cmpHash_np
from tqdm import tqdm
import os
import os.path as osp
import numpy as np
import cv2
import shutil


def similarity(img_dir, remove_dir, threshold=0.95, count_only=False, start=0, end=-1, stype="phash", save=True):
    """Compute the similarity between images and remove these images with high similarity.

    Args:
        img_dir (str): Image dir.
        remove_dir (str): Remove dir.
        threshold (float): The threshold, range [0, 1].
        count_only (bool): Only count how many images will be removed, intead of actually removing them.
        start (int): The index to start with, in case there are too many images, default: 0.
        end (int): The index to end with, in case there are too many images, default: -1.
        stype (str): The calculation type of similarity, could be `phash` and `img`.
        save (bool): There are two modes, save mode and remove mode. Usually the workflow is save(mode) first 
            then remove(mode).
    """
    assert stype in ["phash", "img"]
    compute = pHash if stype == "phash" else img_hash
    compare = cmpHash_np if stype == "phash" else cmpImgHash_np

    if save:
        if osp.exists("hashValue.txt"):
            os.remove("hashValue.txt")
        if osp.exists("hashName.txt"):
            os.remove("hashName.txt")

        imgs_lists = sorted(os.listdir(img_dir))
        pbar = tqdm(imgs_lists, total=len(imgs_lists))
        for p in pbar:
            img = cv2.imread(osp.join(img_dir, p))
            # assert img is not None
            if img is None:
                os.remove(osp.join(img_dir, p))   # remove broken images.
                continue
            with open("hashValue.txt", "a") as fv:
                fv.write(str(compute(img)).replace("[", "").replace("]", "").replace(",", "") + "\n")
            with open("hashName.txt", "a") as fn:
                fn.write(p + "\n")
    else:
        os.makedirs(remove_dir, exist_ok=True)

        with open("hashName.txt", "r") as f:
            names = [i.strip() for i in f.readlines()][start:end]
        values = np.loadtxt("hashValue.txt", dtype=np.uint8)[start:end]
        print(values.shape)
        removed_idx = []

        for i, v in tqdm(enumerate(values), total=len(values)):
            d = compare(v, values)  # distance
            if d.ndim == 2:
                d = d.squeeze(-1)
            s = (64 - d.astype(np.float32)) / 64  # similarity
            if (not (s > threshold).any()) or i in removed_idx:
                continue
            indexes = np.argwhere(s[i + 1 :] > threshold).squeeze(-1) + i + 1
            if len(indexes) == 0:
                continue
            # idx_dir = osp.join(target_dir, f"{i + start}")
            idx_dir = osp.join(remove_dir, f"{names[i]}")
            os.makedirs(idx_dir, exist_ok=True)
            for idx in indexes:
                if idx in removed_idx:
                    continue
                removed_idx.append(idx)
                counter += 1
                if count_only:
                    continue
                shutil.move(osp.join(img_dir, names[idx]), idx_dir)
        print("keep counter:", len(values) - counter)


def generate_fog(img):
    """Generate fake foggy image.

    Args:
        img (np.ndarray): The original image.

    Returns:
        img (np.ndarray): Return foggy image.
        
    """
    exp = 1 if np.random.uniform() < 0.5 else 2
    assert exp in [1, 2]
    h, w = img.shape[:2]
    pixel = np.random.randint(150, 255)
    fog = np.ones((h, w), dtype=np.uint8) * pixel
    start = np.random.uniform(0, 0.2)
    end = np.random.uniform(0.8, 1.0)
    m = np.arange(start, end, step=(end - start)/h)[:h]
    m = m ** 2 if exp == 2 else m
    img = (img * m[:, None, None] + fog[..., None] * (1 - m[:, None, None])).astype(np.uint8)
    return img
