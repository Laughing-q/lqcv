from abc import ABCMeta, abstractmethod
from lqcv.bbox import Boxes
from lqcv.utils.log import LOGGER
from lqcv.utils.plot import plot_one_box, colors
from .utils import IMG_FORMATS
from multiprocessing.pool import Pool
from collections import defaultdict
from tqdm import tqdm
from tabulate import tabulate
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import glob
import os.path as osp
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

    # @abstractmethod
    # def toCOCO(self):
    #     pass
    #
    # @abstractmethod
    # def toXML(self):
    #     pass
    #
    # @abstractmethod
    # def toYOLO(self):
    #     pass

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
                image = cv2.imread(osp.join(self.img_dir, filename))
                if image is None:
                    continue
                cls, bbox = label["cls"], label["bbox"]
                bbox.convert("xyxy")
                for i, c in enumerate(cls):
                    plot_one_box(
                        bbox.data[i],
                        image,
                        color=colors(int(c)),
                        line_thickness=2,
                        label=self.class_names[int(c)],
                    )
            except Exception as e:
                LOGGER.warning(e)
                continue
            if save_dir is not None:
                cv2.imwrite(osp.join(save_dir, filename), image)
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
        assert osp.exists(label_dir), f"The directory '{label_dir}' does not exist."
        if img_dir is None:
            img_dir = label_dir.replace("labels", "images")
            assert osp.exists(
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
                    YOLOConverter.verify_label,
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

    @staticmethod
    def verify_label(args):
        # Verify one image-label pair
        img_name, imgs_dir, labels_dir, num_classes = args
        im_file = osp.join(imgs_dir, img_name)
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
                with open(lb_file, "r") as f:
                    l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    l = np.array(l, dtype=np.float32)
                if len(l):
                    assert l.shape[1] == 5, f"labels require 5 columns each: {lb_file}"
                    assert (l >= 0).all(), f"negative labels: {lb_file}"
                    assert (
                        l[:, 1:] <= 1
                    ).all(), f"non-normalized or out of bounds coordinate labels: {lb_file}"
                    assert np.unique(l, axis=0).shape[0] == l.shape[0], "duplicate labels"
                    assert (
                        l[:, 0] < num_classes
                    ).all(), f"label cls index out of range, {l[:, 0]}, {lb_file}"
                else:
                    ne = 1  # label empty
                    l = np.zeros((0, 5), dtype=np.float32)
            else:
                nm = 1  # label missing
                l = np.zeros((0, 5), dtype=np.float32)
            # NOTE: denormlize coordinates
            l[:, 2::2] *= shape[0]  # h
            l[:, 1::2] *= shape[1]  # w
            return img_name, l, shape, nm, nf, ne, nc, ""
        except Exception as e:
            nc += 1
            msg = f"WARNING: Ignoring corrupted image and/or label {im_file}: {e}"
            return [None, None, None, nm, nf, ne, nc, msg]


class XMLConverter(BaseConverter):
    def __init__(self, label_dir, class_names=None, img_dir=None) -> None:
        """XMLConverter.

        Args:
            label_dir (str): The directory of .xml labels.
            class_names (str | optional): Class names, automatically read names from xml files if not given.
            img_dir (str | optional): Image directory,
                if it's None then assume the structure is like the following example:
                    root/
                    ├── images
                    ├── xmls
        """
        assert osp.exists(label_dir), f"The directory '{label_dir}' does not exist."
        if img_dir is None:
            img_dir = label_dir.replace("xmls", "images")
            assert osp.exists(
                img_dir
            ), f"The directory '{img_dir}' does not exist, please pass `img_dir` arg."
        super().__init__(label_dir, img_dir, class_names)

    def read_labels(self):
        LOGGER.info(f"Read xml labels from {self.label_dir}...")

        catImg = defaultdict(list)
        xml_files = glob.glob(osp.join(self.label_dir, "*.xml"))
        ni = len(xml_files)
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"Scanning '{self.label_dir}' images and labels..."
        with Pool() as pool:
            pbar = tqdm(
                pool.imap_unordered(
                    XMLConverter.verify_label,
                    zip(
                        xml_files,
                        [self.img_dir] * ni,
                        [self.class_names] * ni,
                    ),
                ),
                desc=desc,
                total=len(xml_files),
            )
            for img_name, names, bbox, shape, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if img_name:
                    bbox = Boxes(bbox, format="xyxy")
                    self.labels.append(dict(img_name=img_name, shape=shape, names=names, bbox=bbox))
                    for name in names:
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

        # update class_names
        if self.class_names is None:
            self.class_names = list(self.catCount.keys()) 
        # update cls index
        for l in self.labels:
            names = l.pop("names")
            l["cls"] = [self.class_names.index(n) for n in names]


    @staticmethod
    def verify_label(args):
        xml_file, img_dir, class_names = args
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, corrupt
        try:
            # verify labels
            if os.path.isfile(xml_file):
                nf += 1  # label found
                xml = ET.parse(xml_file).getroot()
                objects = xml.findall("object")
                if len(objects):
                    filename = xml.find("filename")
                    assert (filename is not None), f"can't get `filename` info from {xml_file}"
                    filename = str(filename.text)
                    # filename = Path(xml_file).name
                    try:
                        size = xml.find("size")
                        width = size.find("width")
                        height = size.find("height")
                        h = int(height.text)
                        w = int(width.text)
                    except:
                        assert (
                            img_dir is not None
                        ), f"can't get `width` or `height` info from {xml_file}"
                        img_path = osp.join(img_dir, filename)
                        assert osp.exists(img_path), \
                                f"can't get `width` or `height` info from {xml_file} also can't find img file {img_path}"
                        h, w = cv2.imread(img_path).shape[:2]
                    shape = (h, w, 3)
                    cls, bbox = [], []
                    for obj in objects:
                        name = obj.find("name").text
                        if class_names is not None:
                            assert (
                                name in class_names
                            ), f"'{name}' not in {class_names} from {xml_file}"
                        cls.append(name)
                        box = obj.find("bndbox")
                        x1 = float(box.find("xmin").text)
                        x2 = float(box.find("xmax").text)
                        y1 = float(box.find("ymin").text)
                        y2 = float(box.find("ymax").text)
                        bbox.append([x1, y1, x2, y2])
                    bbox = np.array(bbox, dtype=np.float32)
                    assert (bbox >= 0).all(), f"negative labels: {xml_file}"
                    assert (bbox[:, 0::2] <= w).all() and (bbox[:, 1::2] <= h).all(), \
                            f"non-normalized or out of bounds coordinate labels: {xml_file}"
                    assert np.unique(bbox, axis=0).shape[0] == bbox.shape[0], "duplicate labels"
                else:
                    ne += 1
                    cls, bbox = [], np.zeros((0, 4), dtype=np.float32)
            else:
                nm += 1
                cls, bbox = [], np.zeros((0, 4), dtype=np.float32)
            return filename, cls, bbox, shape, nm, nf, ne, nc, ""
        except Exception as e:
            print(e)
            nc += 1
            msg = f"WARNING: Ignoring corrupted xmls: {e}"
            return [None, None, None, None, nm, nf, ne, nc, msg]
