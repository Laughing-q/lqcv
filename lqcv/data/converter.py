from abc import ABCMeta, abstractmethod
from lqcv.utils.log import LOGGER
from multiprocessing.pool import Pool
from collections import defaultdict
from tqdm import tqdm
import os
from pathlib import Path


class BaseConverter(metaclass=ABCMeta):
    def __init__(self, label_dir, img_dir=None, class_names=None) -> None:
        super().__init__()
        self.imgToAnns = {}
        self.catCount = {}
        self.catImgCnt = {}
        self.imgs_wh = {}
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.class_names = class_names

    @abstractmethod
    def toCOCO(self):
        pass

    @abstractmethod
    def toXML(self):
        pass

    @abstractmethod
    def toYOLO(self):
        pass

    @abstractmethod
    def read_labels(self):
        pass

    def visualize(self):
        pass


class YOLOConverter(BaseConverter):
    def __init__(self, label_dir, class_names, img_dir=None) -> None:
        """YOLOConverter.

        Args:
            label_dir (str): The directory of .txt labels.
            class_names (str): Class names.
            img_dir (str | optional): Image directory, 
                if it's None then assume the structure is like the following example:
                    root/
                    ├── images
                    ├── labels
        """
        assert os.path.exists(label_dir), f"The directory '{label_dir}' does not exist."
        if img_dir is None:
            img_dir = label_dir.replace("labels", "images")
            assert os.path.exists(img_dir), f"The directory '{img_dir}' does not exist, please pass `img_dir` arg."
        super().__init__(label_dir, img_dir, class_names)

    def read_labels(self):
        LOGGER.info(f"Read yolo labels from {self.labels_dir}...")
        imgs_wh = {}
        imgToAnns, catImgCnt, catCount = defaultdict(list), defaultdict(list), defaultdict(int)

        img_names = os.listdir(self.imgs_dir)
        ni = len(img_names)
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"Scanning '{self.labels_dir}' images and labels..."
        with Pool() as pool:
            pbar = tqdm(
                pool.imap_unordered(
                    self.verify_image_label,
                    zip(
                        img_names,
                        [self.imgs_dir] * ni,
                        [self.labels_dir] * ni,
                        [len(self.class_names)] * ni,
                    ),
                ),
                desc=desc,
                total=len(img_names),
            )
            for img_name, label, shape, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if img_name:
                    imgs_wh[img_name] = {"width": shape[0], "height": shape[1], "channel": shape[2]}
                if label is not None:
                    for l in label:
                        category_id = int(l[0])
                        bbox = l[1:]
                        name = self.class_names[int(category_id)]
                        imgToAnns[img_name].append(
                            {
                                "class_name": name,
                                "bbox": bbox,
                            }
                        )
                        catCount[name] += 1
                        catImgCnt[name].append(img_name)
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        return imgToAnns, catImgCnt, catCount, imgs_wh
