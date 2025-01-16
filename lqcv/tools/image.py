from lqcv.simi.hash import img_hash, cmpImgHash_np, pHash, cmpHash_np
from lqcv.utils.log import LOGGER
from pathlib import Path
from tqdm import tqdm
import os
import os.path as osp
import numpy as np
import cv2
import shutil


def similarity(img_dir, threshold=0.95, count_only=False, stype="phash", name=""):
    """Compute the similarity between images and remove these images with high similarity.

    Args:
        img_dir (str): Image dir.
        threshold (float): The threshold, range [0, 1].
        count_only (bool): Only count how many images will be removed, intead of actually removing them.
        stype (str): The calculation type of similarity, could be `phash` and `img`.
        name (str): The save name for HashValues and HashNames.
    """
    assert stype in ["phash", "img"]
    compute = pHash if stype == "phash" else img_hash
    compare = cmpHash_np if stype == "phash" else cmpImgHash_np

    def _create_values(filename_path, value_path):
        """Create new hash map."""
        if osp.exists(filename_path):
            os.remove(filename_path)
        if osp.exists(value_path):
            os.remove(value_path)
        imgs_lists = os.listdir(img_dir)
        pbar = tqdm(imgs_lists, total=len(imgs_lists), desc="Creating values")
        for p in pbar:
            img = cv2.imread(osp.join(img_dir, p))
            # assert img is not None
            if img is None:
                os.remove(osp.join(img_dir, p))  # remove broken images.
                continue
            with open(value_path, "a") as fv:
                fv.write(str(compute(img)).replace("[", "").replace("]", "").replace(",", "") + "\n")
            with open(filename_path, "a") as fn:
                fn.write(p + "\n")

    im_parent = Path(img_dir).parent
    filename = im_parent / f"{name}Name.txt"
    value = im_parent / f"{name}Value.txt"
    if not (filename.exists() and value.exists()):
        LOGGER.info("Can't find Name.txt or Value.txt! Creating automatically!")
        if count_only == False:
            LOGGER.info("Force to set `count_only=True` for the first time!")
            count_only = True  # force to set count_only=True for the first time.
        _create_values(str(filename), str(value))

    im_names = os.listdir(img_dir)
    with open(str(filename), "r") as f:
        names = [i.strip() for i in f.readlines()]
    values = np.loadtxt(str(value), dtype=np.uint8)
    if len(names) != len(values) or names != im_names:
        LOGGER.info("`names` is not matched! Creating new one automatically!")
        if count_only == False:
            LOGGER.info("Force to set `count_only=True` for the first time!")
            count_only = True  # force to set count_only=True for the first time.
        _create_values(str(filename), str(value))
        # reload names and values
        with open(str(filename), "r") as f:
            names = [i.strip() for i in f.readlines()]
        values = np.loadtxt(str(value), dtype=np.uint8)

    remove_dir = im_parent / "removed"
    remove_dir.mkdir(parents=True, exist_ok=True)
    removed_idx = []
    counter = 0

    pbar = tqdm(enumerate(values), total=len(values))
    pbar.desc = f"{name}"
    for i, v in pbar:
        d = compare(v, values)  # distance
        if d.ndim == 2:
            d = d.squeeze(-1)
        s = (64 - d.astype(np.float32)) / 64  # similarity
        if (not (s > threshold).any()) or i in removed_idx:
            continue
        indexes = np.argwhere(s[i + 1 :] > threshold).squeeze(-1) + i + 1
        if len(indexes) == 0:
            continue
        idx_dir = osp.join(remove_dir, f"{names[i]}")
        for idx in indexes:
            if idx in removed_idx:
                continue
            removed_idx.append(idx)
            counter += 1
            if count_only:
                continue
            os.makedirs(idx_dir, exist_ok=True)
            shutil.move(osp.join(img_dir, names[idx]), idx_dir)
    print(
        f"keep counter:{len(values) - counter}/{len(values)}",
    )


def similarity_yolo(img_dir, threshold=0.95, count_only=False, model="yolo11n-cls.pt", name="", gpu=True):
    """Compute the similarity between images and remove these images with high similarity, powered by YOLO models.

    Args:
        img_dir (str): Image dir.
        threshold (float | List[float]): The threshold, range [0, 1].
        count_only (bool): Only count how many images will be removed, intead of actually removing them.
        model (str): The model using to calculate the embeddings.
        name (str): The save name for HashValues and HashNames.
        gpu (bool): Whether to use gpu to calculate the embeddings.
    """

    import torch

    def _create_values(filename_path, value_path, model):
        """Create new hash map."""
        from ultralytics import YOLO

        model = YOLO(model)
        if osp.exists(filename_path):
            os.remove(filename_path)
        if osp.exists(value_path):
            os.remove(value_path)
        imgs_lists = os.listdir(img_dir)
        pbar = tqdm(imgs_lists, total=len(imgs_lists), desc="Creating values")
        feats = []
        for p in pbar:
            img = cv2.imread(osp.join(img_dir, p))
            # assert img is not None
            if img is None:
                os.remove(osp.join(img_dir, p))  # remove broken images.
                continue

            feat = model.predict(img, embed=True, verbose=False, half=True)[0]  # (1, 1280)
            feat = torch.nn.functional.normalize(feat if gpu else feat.cpu())
            feats.append(feat)
            with open(filename_path, "a") as fn:
                fn.write(p + "\n")
        torch.save(torch.cat(feats, dim=0), value_path)  # (N, 1280)

    if isinstance(threshold, list):
        assert count_only, f"Expected `count_only=True` when passing a list of threshold."
    if not isinstance(threshold, list):
        threshold = [threshold]

    im_parent = Path(img_dir).parent
    filename = im_parent / f"{name}Name.txt"
    value = im_parent / f"{name}Value.pth"
    if not (filename.exists() and value.exists()):
        LOGGER.info("Can't find Name.txt or Value.txt! Creating automatically!")
        if count_only == False:
            LOGGER.info("Force to set `count_only=True` for the first time!")
            count_only = True  # force to set count_only=True for the first time.
        _create_values(str(filename), str(value), model=model)

    im_names = os.listdir(img_dir)
    with open(str(filename), "r") as f:
        names = [i.strip() for i in f.readlines()]
    values = torch.load(str(value))  # (N, 1280)
    if len(names) != len(values) or names != im_names:
        LOGGER.info("`names` is not matched! Creating new one automatically!")
        if count_only == False:
            LOGGER.info("Force to set `count_only=True` for the first time!")
            count_only = True  # force to set count_only=True for the first time.
        _create_values(str(filename), str(value), model=model)
        # reload names and values
        with open(str(filename), "r") as f:
            names = [i.strip() for i in f.readlines()]
        values = torch.load(str(value))  # (N, 1280)

    remove_dir = im_parent / "removed"
    remove_dir.mkdir(parents=True, exist_ok=True)
    # NOTE: [[]] * len(threshold) will have the same reference.
    removed_idx = [[] for _ in range(len(threshold))]
    counter = [0] * len(threshold)

    values = values.cuda() if gpu else values  # in case the saved `values` are loaded on cpu
    pbar = tqdm(enumerate(values), total=len(values))
    pbar.desc = f"{name}"
    for i, v in pbar:
        # s = np.dot(v, values.T)
        # NOTE: using pytorch is way more faster than np.dot(even running on cpu)
        s = torch.mm(v.unsqueeze(0), values.T).cpu().numpy().squeeze(0)
        for ti, t in enumerate(threshold):
            if (not (s > t).any()) or i in removed_idx[ti]:
                continue
            indexes = np.argwhere(s[i + 1 :] > t).squeeze(-1) + i + 1
            if len(indexes) == 0:
                continue
            idx_dir = osp.join(remove_dir, f"{names[i]}")
            for idx in indexes:
                if idx in removed_idx[ti]:
                    continue
                removed_idx[ti].append(idx)
                counter[ti] += 1
                if count_only:
                    continue
                os.makedirs(idx_dir, exist_ok=True)
                shutil.move(osp.join(img_dir, names[idx]), idx_dir)
    for count in counter:
        print(
            f"keep counter:{len(values) - count}/{len(values)}",
        )


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
    m = np.arange(start, end, step=(end - start) / h)[:h]
    m = m**2 if exp == 2 else m
    img = (img * m[:, None, None] + fog[..., None] * (1 - m[:, None, None])).astype(np.uint8)
    return img


def resize(im, max_size, scaleup=False):
    """Resize image to max size while keeping aspect ratio.

    Args:
        im (np.ndarray): The original image.
        max_size (int): Max size of the shape.
        scaleup (bool): Whether to resize up image if the original shape is small than the max_size.
            default: False.
    Returns:
        The resized image with max_size.
    """
    shape = im.shape[:2]
    r = min(max_size / shape[0], max_size / shape[1])
    if (r >= 1.0) and not scaleup:
        return im
    new_shape = int(round(shape[1] * r)), int(round(shape[0] * r))  # w, h
    im = cv2.resize(im, new_shape)
    return im
