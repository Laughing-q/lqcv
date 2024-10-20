from .yolo import YOLOConverter
from lqcv.bbox import Boxes
from lqcv.utils.log import LOGGER
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import os.path as osp
import os
import cv2


class XMLConverter(YOLOConverter):
    def __init__(self, label_dir, class_names=None, img_dir=None, chunk_size=None) -> None:
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
        if img_dir is None:
            img_dir = label_dir.replace("xmls", "images")
        super().__init__(label_dir, class_names, img_dir, chunk_size)
        self.format = 'xml'

    def read_labels(self, label_dir, chunk_size=None):
        super().read_labels(label_dir, chunk_size)
        # update class_names
        if self.class_names is None:
            self.class_names = list(self.catCount.keys()) 
            LOGGER.warning("No class names provided, reading it automatically from xml files!")
        # update cls names cls indexs
        for l in self.labels:
            names = l.pop("cls")
            l["cls"] = [self.class_names.index(n) for n in names]

    @classmethod
    def verify_label(cls, args):
        img_name, img_dir, labels_dir, class_names = args
        xml_file = osp.join(labels_dir, str(Path(img_name).with_suffix(".xml")))
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, corrupt
        filename, shape = None, None
        try:
            # verify labels
            if os.path.isfile(xml_file):
                nf += 1  # label found
                xml = ET.parse(xml_file).getroot()
                objects = xml.findall("object")
                nl = len(objects)
                if nl:
                    filename = xml.find("filename")
                    assert (filename is not None), f"can't get `filename` info from {xml_file}"
                    filename = str(filename.text)
                    if filename != img_name:
                        # LOGGER.warning("[Filename] filename got different name from image name, "
                        #         f"'{filename}' vs '{img_name}' using image name(the latter)!")
                        filename = img_name
                    try:
                        size = xml.find("size")
                        width = size.find("width")
                        height = size.find("height")
                        h = int(height.text)
                        w = int(width.text)
                    except:
                        assert (img_dir is not None), f"can't get `width` or `height` info from {xml_file}"
                        img_path = osp.join(img_dir, filename)
                        assert osp.exists(img_path), \
                                f"can't get `width` or `height` info from {xml_file} also can't find img file {img_path}"
                        h, w = cv2.imread(img_path).shape[:2]
                    shape = (h, w, 3)
                    cls, bbox = [], []
                    for obj in objects:
                        name = obj.find("name").text
                        if class_names is not None:
                            assert (name in class_names), \
                                    f"'{name}' not in {class_names} from {xml_file}"
                        cls.append(name)
                        box = obj.find("bndbox")
                        x1 = min(max(0, float(box.find("xmin").text)), w)
                        x2 = min(max(0, float(box.find("xmax").text)), w)
                        y1 = min(max(0, float(box.find("ymin").text)), h)
                        y2 = min(max(0, float(box.find("ymax").text)), h)
                        bbox.append([x1, y1, x2, y2])
                    bbox = np.array(bbox, dtype=np.float32)
                    assert (bbox >= 0).all(), f"negative labels: {xml_file}"
                    assert (bbox[:, 0::2] <= w).all() and (bbox[:, 1::2] <= h).all(), \
                            f"non-normalized or out of bounds coordinate labels: {xml_file}"
                    _, idx = np.unique(bbox, axis=0, return_index=True)
                    if len(idx) < nl:  # duplicate row check
                        bbox = bbox[idx]  # remove duplicates
                        cls = [cls[ii] for ii in idx]
                        msg = f'WARNING ⚠️ {xml_file}: {nl - len(idx)} duplicate labels removed'
                else:
                    ne += 1
                    cls, bbox = [], np.zeros((0, 4), dtype=np.float32)
            else:
                nm += 1
                cls, bbox = [], np.zeros((0, 4), dtype=np.float32)
            return filename, cls, Boxes(bbox, format="xyxy"), shape, nm, nf, ne, nc, ""
        except Exception as e:
            nc += 1
            msg = f"WARNING: Ignoring corrupted xml {xml_file}: {e}"
            return [None, None, None, None, nm, nf, ne, nc, msg]
