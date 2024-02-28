from lqcv.utils.log import LOGGER
from lqcv.bbox import Boxes
from .base import BaseConverter
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
import os.path as osp


class COCOConverter(BaseConverter):
    def __init__(self, json_file, img_dir=None) -> None:
        """COCOConverter.

        Args:
            json_file (str): The json file of coco.
            img_dir (str | optional): Image directory,
                if it's None then assume the structure is like the following example:
                    root/
                    ├── images
                    ├── json_file
        """
        if img_dir is None:
            img_dir = Path(json_file).parent / "images"
        if not osp.exists(img_dir):
            LOGGER.warning(f"The directory '{img_dir}' does not exist, `visualize` is not available.")
        self.coco_idx_map = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
                             None, 24, 25, None, None, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, None, 40,
                             41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, None, 60, None,
                             None, 61, None, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, None, 73, 74, 75, 76, 77, 78, 
                             79, None]
        super().__init__(json_file, img_dir=img_dir)
        self.format = 'coco'

    def read_labels(self, json_file):
        is_coco = True if "train2017" in json_file or "val2017" in json_file else False
        with open(json_file, "r") as f:
            data = json.load(f)
        # 'images', 'annotations', 'categories'
        images = {'%g' % x['id']: x for x in data['images']}
        imgToAnns, catImg = defaultdict(list), defaultdict(list)
        # add class_names
        self.class_names = [item["name"] for item in data["categories"]]

        for ann in data['annotations']:
            imgToAnns[ann['image_id']].append(ann)
        ne, nf = 0, 0   # num empty
        pbar = tqdm(imgToAnns.items())
        for img_id, anns in pbar:
            img = images['%g' % img_id]
            h, w, f = img['height'], img['width'], img['file_name']
            cls, bbox = [], []
            for ann in anns:
                if ann.get('iscrowd', False):  # ignore iscrowd
                    continue
                cls.append(self.coco_idx_map[ann["category_id"] - 1] if is_coco else ann["category_id"] - 1)
                bbox.append(ann["bbox"])
            ne += (0 if len(cls) else 1)
            nf += (1 if len(cls) else 0)
            bbox = np.array(bbox, dtype=np.float32) if len(bbox) else np.zeros((0, 4), dtype=np.float32)
            self.labels.append(dict(img_name=f, shape=(h, w, 3), cls=cls, bbox=Boxes(bbox, format="ltwh")))

            for c in cls:
                name = self.class_names[int(c)]
                self.catCount[name] += 1
                if f not in catImg[name]:
                    catImg[name].append(f)

            pbar.desc = f"Annotations {json_file}...{nf} found, {ne} empty"
        # update catImgCnt
        for name, imgCnt in catImg.items():
            self.catImgCnt[name] = len(imgCnt)
