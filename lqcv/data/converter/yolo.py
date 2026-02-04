from ultralytics.utils.ops import segments2boxes
from lqcv.bbox import Boxes
from .base import BaseConverter
from lqcv.utils.log import LOGGER
from ..utils import IMG_FORMATS
from multiprocessing.pool import Pool
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import numpy as np
import os.path as osp
import os
import cv2


class YOLOConverter(BaseConverter):
    def __init__(self, label_dir, class_names=None, img_dir=None, chunk_size=None) -> None:
        """YOLOConverter.

        Args:
            label_dir (str): The directory of .txt labels.
            class_names (List[str] | optional): Class names.
            img_dir (str | optional): Image directory,
                if it's None then assume the structure is like the following example:
                    root/
                    ├── images
                    ├── labels
        """
        if img_dir is None:
            img_dir = label_dir.replace("labels", "images")
        if self.__class__.__name__ == "YOLOConverter":  # Do not affect XMLConverter
            assert osp.exists(img_dir), f"The directory '{img_dir}' does not exist, please pass `img_dir` arg."
        super().__init__(label_dir, class_names, img_dir, chunk_size)
        self.format = "yolo"

    def read_labels(self, label_dir, chunk_size=None):
        """Read labels.

        Args:
            label_dir (str):
            chunk_size (int, optional):
        """
        LOGGER.info(f"Read labels from {label_dir}...")

        catImg = defaultdict(list)
        img_names = os.listdir(label_dir if self.img_dir is None or not Path(self.img_dir).exists() else self.img_dir)
        img_names = (
            [img_names[i : i + chunk_size] for i in range(0, len(img_names), chunk_size)]
            if chunk_size != None
            else [img_names]
        )

        for im_names in img_names:
            ni = len(im_names)
            nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
            desc = f"Scanning '{label_dir}' images and labels..."
            with Pool() as pool:
                pbar = tqdm(
                    pool.imap_unordered(
                        self.verify_label,
                        zip(
                            im_names,
                            [self.img_dir] * ni,
                            [label_dir] * ni,
                            [self.class_names] * ni,
                        ),
                    ),
                    desc=desc,
                    total=len(im_names),
                )
                for img_name, cls, bbox, shape, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                    nm += nm_f
                    nf += nf_f
                    ne += ne_f
                    nc += nc_f
                    if img_name:
                        self.labels.append(dict(img_name=img_name, shape=shape, cls=cls, bbox=bbox))
                        for c in cls:
                            name = c if self.class_names is None or isinstance(c, str) else self.class_names[int(c)]
                            self.catCount[name] += 1
                            if img_name not in catImg[name]:
                                catImg[name].append(img_name)
                    # TODO: self.errors
                    if msg:
                        LOGGER.warning(msg)
                        msgs.append(msg)
                    pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        # update catImgCnt
        for name, imgCnt in catImg.items():
            self.catImgCnt[name] = len(imgCnt)
        if self.class_names is None:  # is no class names provided
            self.class_names = list(range(int(max(self.catCount.keys())) + 1))

    @classmethod
    def verify_label(self, args):
        # Verify one image-label pair
        img_name, img_dir, labels_dir, class_names = args
        im_file = osp.join(img_dir, img_name)
        lb_file = osp.join(labels_dir, str(Path(img_name).with_suffix(".txt")))
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, corrupt
        try:
            # verify images
            im = cv2.imread(im_file)
            shape = im.shape  # h, w, c
            suffix = Path(im_file).suffix[1:]
            assert suffix.lower() in IMG_FORMATS, f"invalid image format {suffix}"

            # verify labels
            if os.path.isfile(lb_file):
                nf = 1  # label found
                with open(lb_file) as f:
                    l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    if any(len(x) > 6 for x in l):  # is segment
                        classes = np.array([x[0] for x in l], dtype=np.float32)
                        segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]  # (cls, xy1...)
                        l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                l = np.array(l, dtype=np.float32)
                nl = len(l)
                if len(l):
                    assert l.shape[1] == 5, f"labels require 5 columns each: {lb_file}"
                    assert (l >= 0).all(), f"negative labels: {lb_file}"
                    assert (l[:, 1:] <= 1).all(), f"non-normalized or out of bounds coordinate labels: {lb_file}"
                    _, i = np.unique(l, axis=0, return_index=True)
                    if len(i) < nl:  # duplicate row check
                        l = l[i]  # remove duplicates
                        msg = f"WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed"
                    assert np.unique(l, axis=0).shape[0] == l.shape[0], "duplicate labels"
                    if class_names is not None:
                        assert (l[:, 0] < len(class_names)).all(), f"label cls index out of range, {l[:, 0]}, {lb_file}"
                else:
                    ne = 1  # label empty
                    l = np.zeros((0, 5), dtype=np.float32)
            else:
                nm = 1  # label missing
                l = np.zeros((0, 5), dtype=np.float32)
            # NOTE: denormlize coordinates
            l[:, 2::2] *= shape[0]  # h
            l[:, 1::2] *= shape[1]  # w
            return img_name, l[:, 0].tolist(), Boxes(l[:, 1:], format="xywh"), shape, nm, nf, ne, nc, ""
        except Exception as e:
            nc += 1
            msg = f"WARNING: Ignoring corrupted image and/or label {im_file}: {e}"
            return [None, None, None, None, nm, nf, ne, nc, msg]
