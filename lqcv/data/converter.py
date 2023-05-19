from abc import ABCMeta, abstractmethod
from lqcv.bbox import Boxes
from lqcv.utils.log import LOGGER
from lqcv.utils.plot import plot_one_box, colors
from .utils import verify_image_label
from multiprocessing.pool import Pool
from collections import defaultdict
from tqdm import tqdm
from tabulate import tabulate
import numpy as np
import os
import cv2


class BaseConverter(metaclass=ABCMeta):
    def __init__(self, label_dir, img_dir=None, class_names=None) -> None:
        super().__init__()
        self.labels = list()
        self.catCount = defaultdict(int)
        self.catImgCnt = dict()
        self.imgs_wh = dict()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.class_names = class_names

        self.read_labels()

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

    def visualize(self, save_dir=None):
        if self.img_dir is None:
            LOGGER.warning("`self.img_dir` is None.")
            return

        pbar = tqdm(self.labels, total=len(self.labels))
        if save_dir is None:
            cv2.namedWindow("p", cv2.WINDOW_NORMAL)
        for label in pbar:
            try:
                filename = label["img_name"]
                image = cv2.imread(os.path.join(self.img_dir, filename))
                if image is None:
                    continue
                cls, bbox = label["cls"], label["bbox"]
                bbox.convert("xyxy")
                for i, c in enumerate(cls):
                    plot_one_box(bbox.data[i], image, color=colors(int(c)), line_thickness=2, label=None)
            except Exception as e:
                LOGGER.warning(e)
                continue
            if save_dir is not None:
                cv2.imwrite(os.path.join(save_dir, filename), image)
            else:
                cv2.imshow("p", image)
                if cv2.waitKey(0) == ord("q"):
                    break

    def __repr__(self):
        total_img = 0
        total_obj = 0
        cat_table = []
        for i, c in enumerate(self.class_names):
            if c in self.catImgCnt and c in self.catCount:
                total_img += self.catImgCnt[c]
                total_obj += self.catCount[c]
                cat_table.append((str(i), str(c), self.catImgCnt[c], self.catCount[c]))
            else:
                cat_table.append((str(i), str(c), 0, 0))

        cat_table += [(" ", "total", total_img, total_obj)]
        return "\n" + tabulate(
            cat_table,
            headers=["Id", "Category", "ImageCnt", "ClassCnt"],
            tablefmt="fancy_grid",
            missingval="None",
        )


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
            assert os.path.exists(
                img_dir
            ), f"The directory '{img_dir}' does not exist, please pass `img_dir` arg."
        super().__init__(label_dir, img_dir, class_names)

    def read_labels(self):
        LOGGER.info(f"Read yolo labels from {self.label_dir}...")

        catImg = defaultdict(list)
        img_names = os.listdir(self.img_dir)
        ni = len(img_names)
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"Scanning '{self.label_dir}' images and labels..."
        with Pool() as pool:
            pbar = tqdm(
                pool.imap_unordered(
                    verify_image_label,
                    zip(
                        img_names,
                        [self.img_dir] * ni,
                        [self.label_dir] * ni,
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
                    cls = label[:, 0]
                    bbox = Boxes(label[:, 1:], format="xywh")
                    self.labels.append(dict(img_name=img_name, shape=shape, cls=cls, bbox=bbox))
                    for c in cls:
                        name = self.class_names[int(c)]
                        self.catCount[name] += 1
                        if img_name not in catImg[name]:
                            catImg[name].append(img_name)
                # TODO: self.errors
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        # update catImgCnt
        for name, imgCnt in catImg.items():
            self.catImgCnt[name] = len(imgCnt)

    def toCOCO(self):
        pass

    def toXML(self):
        pass

    def toYOLO(self):
        pass
