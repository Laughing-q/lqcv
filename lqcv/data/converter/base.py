from abc import ABCMeta, abstractmethod
from lqcv.utils.log import LOGGER
from lqcv.bbox import Boxes
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from tabulate import tabulate
import os.path as osp
import numpy as np
import os
import cv2
import shutil
import json



class BaseConverter(metaclass=ABCMeta):
    def __init__(self, label_dir, class_names=None, img_dir=None) -> None:
        super().__init__()
        assert osp.exists(label_dir), f"The directory/file '{label_dir}' does not exist."

        self.labels = list()
        self.catCount = defaultdict(int)
        self.catImgCnt = dict()
        self.img_dir = img_dir
        self.class_names = class_names

        self.read_labels(label_dir)

    def toCOCO(self, 
               save_file, 
               classes=None, 
               im_dir=None):
        """Convert labels to coco format.

        Args:
            save_file (str): Save path of the dst json file.
            classes (Optiona | List[str]): Filter the class if given.
            im_dir (Optional | str): Move the images to im_dir if given and `classes` is also given.
        """
        if self.format == "coco" and classes is None:
            LOGGER.info("Current format is COCO! there's no need to convert it since `classes` is also `None`.")
            return
        class_name = classes if classes is not None else self.class_names
        cocoDict = dict()
        images = list()
        annotations = list()
        objid = 1
        copy_im = im_dir is not None and classes is not None and self.img_dir is not None

        pbar = tqdm(enumerate(self.labels), total=len(self.labels))
        pbar.desc = f"Convert {self.format.upper()} to COCO: "
        for idx, label in pbar:
            h, w = label["shape"][:2]
            image, sub_annotations = dict(), []
            image['file_name'] = label["img_name"]

            image['height'] = h
            image['width'] = w
            image['id'] = idx

            cls, bboxes = label["cls"], label["bbox"]
            bboxes.convert("ltwh")
            for i, c in enumerate(cls):
                name = self.class_names[int(c)]
                if name not in class_name:
                    LOGGER.info(f"`{name}` not in {class_name}, ignore")
                    continue
                category_id = class_name.index(name)

                annotation = dict()
                annotation["image_id"] = idx
                annotation["ignore"] = 0
                annotation["iscrowd"] = 0
                # xyxy -> tlwh
                x, y, w, h = bboxes[i].data.squeeze().tolist()

                annotation["bbox"] = [x, y, w, h]
                annotation["area"] = float(w * h)
                annotation["category_id"] = category_id
                annotation["id"] = objid
                objid += 1
                annotation["segmentation"] = [[x, y, x, (y + h), (x + w), (y + h), (x + w), y]]
                sub_annotations.append(annotation)

            if len(sub_annotations):
                images.append(image)
                annotations += sub_annotations
                if copy_im:
                    shutil.copy(osp.join(self.img_dir, label["img_name"]), im_dir)

        cocoDict["images"] = images
        cocoDict["annotations"] = annotations
        cocoDict["categories"] = [{"supercategory": "none", "id": c, 
                                   "name": class_name[c]} for c in range(len(class_name))]
        cocoDict["type"] = "instances"

        # print attrDict
        Path(save_file).parent.mkdir(parents=True, exist_ok=True)
        jsonString = json.dumps(cocoDict, indent=2)
        with open(save_file, "w") as f:
            f.write(jsonString)

        LOGGER.info(f"Convert results: {len(images)}/{len(self.labels)}")

    def toXML(self, 
               save_dir, 
               classes=None, 
               im_dir=None):
        """Convert labels to xml format.

        Args:
            save_dir (str): Save dir for the dst xml files.
            classes (Optiona | List[str]): Filter the class if given.
            im_dir (Optional | str): Move the images to im_dir if given and `classes` is also given.
        """
        if self.format == "xml" and classes is None:
            LOGGER.info("Current format is XML! there's no need to convert it since `classes` is also `None`.")
            return
        class_name = classes if classes is not None else self.class_names
        os.makedirs(save_dir, exist_ok=True)
        copy_im = im_dir is not None and classes is not None and self.img_dir is not None

        anno_temp, obj_temp = self.get_xml_template()
        pbar = tqdm(self.labels, total=len(self.labels))
        pbar.desc = f"Convert {self.format.upper()} to XML: "
        for label in pbar:
            h, w, c = label["shape"]
            cls, bboxes = label["cls"], label["bbox"]
            bboxes.convert("xyxy")
            filename = label["img_name"]

            obj_str = ''
            xml_name = str(Path(filename).with_suffix('.xml'))
            xml_path = osp.join(save_dir, xml_name)
            for i, c in enumerate(cls):
                name = self.class_names[int(c)]

                if name not in class_name:
                    LOGGER.info(f"`{name}` not in {class_name}, ignore")
                    continue

                bbox = bboxes[i].data.squeeze().tolist()
                obj_str += obj_temp % (name, *bbox)
            if len(obj_str):
                f_xml = open(xml_path, 'w')
                f_xml.write(anno_temp % (filename, w, h, c, obj_str))
                f_xml.close()
                if copy_im:
                    shutil.copy(osp.join(self.img_dir, label["img_name"]), im_dir)

        LOGGER.info(f"Convert results: {len(os.listdir(save_dir))}/{len(self.labels)}")

    def toYOLO(self, 
               save_dir, 
               classes=None, 
               im_dir=None,
               single_cls=False):
        """Convert labels to yolo format.

        Args:
            save_dir (str): Save dir for the dst txt files.
            classes (Optional | List[str]): Filter the class if given.
            im_dir (Optional | str): Move the images to im_dir if given and `classes` is also given.
            single_cls (Optional | bool): Whether to treat all the classes to one class, default: False.
        """
        if self.format == "yolo" and classes is None:
            LOGGER.info("Current format is YOLO! there's no need to convert it since `classes` is also `None`.")
            return
        class_name = classes if classes is not None else self.class_names
        os.makedirs(save_dir, exist_ok=True)
        copy_im = im_dir is not None and classes is not None and self.img_dir is not None
        if copy_im:
            os.makedirs(im_dir, exist_ok=True)

        pbar = tqdm(self.labels, total=len(self.labels))
        pbar.desc = f"Convert {self.format.upper()} to YOLO: "
        for label in pbar:
            h, w = label["shape"][:2]
            cls, bboxes = label["cls"], label["bbox"]
            bboxes.convert("xywh")
            filename = label["img_name"]

            label_name = str(Path(filename).with_suffix('.txt'))
            label_path = osp.join(save_dir, label_name)

            label = ""
            for i, c in enumerate(cls):
                name = self.class_names[int(c)]

                if name not in class_name:
                    LOGGER.info(f"`{name}` not in {class_name}, ignore")
                    continue

                cx, cy, bw, bh = bboxes[i].data.squeeze().tolist()
                category_id = 0 if single_cls else class_name.index(name)
                cx /= w
                cy /= h
                bw /= w
                bh /= h

                label += "%s %s %s %s %s\n" % (category_id, cx, cy, bw, bh)

            if len(label):
                f_label = open(label_path, 'w')
                f_label.write(label)
                f_label.close()
                if copy_im:
                    shutil.copy(osp.join(self.img_dir, filename), im_dir)

        LOGGER.info(f"Convert results: {len(os.listdir(save_dir))}/{len(self.labels)}")


    def check(self, iou_thres=0.7, min_pixel=5, filter=False):
        """Check dataset.

        Args:
            iou_thres (float): The iou threshold.
            min_pixel (int): The minimal pixel for width and height.
            filter (bool): Whether to filter these boxes that is too small or overlaps too much.
        """
        assert len(self.labels), "Checking process needs labels, No labels detected!"
        pbar = tqdm(self.labels, total=len(self.labels))
        pbar.desc = f"Checking dataset: "
        for label in pbar:
            label["sign"] = {}
            filename = label["img_name"]
            bboxes = label["bbox"]
            bboxes.convert("xywh")
            # Init pick indices
            pixel_pick = iou_pick = np.ones(len(bboxes), dtype=bool)
            pixel_idx = np.nonzero((bboxes.data[:, 2:] < min_pixel).any(1))[0]
            if len(pixel_idx):
                LOGGER.warning(f"WARNING ⚠️{filename} got boxes with width or height less than {min_pixel}!")
                # NOTE: no need to keep indices if filter=True, as the indices would be incorrect anyway.
                label["sign"]["pixel"] = pixel_idx if not filter else list(range(len(bboxes)))
                pixel_pick = (bboxes.data[:, 2:] >= min_pixel).any(1)
            iou = np.triu(Boxes.iou(bboxes, bboxes), k=1)
            iou_idx = np.unique(np.concatenate(np.nonzero((iou > iou_thres))))
            if len(iou_idx):
                LOGGER.warning(f"WARNING ⚠️{filename} got boxes with overlaps greater than {iou_thres}!")
                # NOTE: no need to keep indices if filter=True, as the indices would be incorrect anyway.
                label["sign"]["overlap"] = iou_idx if not filter else list(range(len(bboxes)))
                iou_pick = iou.max(axis=0) < iou_thres
            if filter and (len(iou_idx) or len(pixel_idx)):
                ori_len = len(bboxes)
                pick = iou_pick & pixel_pick
                label["bbox"] = bboxes[pick]
                label["cls"] = [c for i, c in enumerate(label["cls"]) if pick[i]]
                LOGGER.info(f"Filter results: {len(label['bbox'])}/{ori_len}")


    @abstractmethod
    def read_labels(self, label_dir):
        pass

    def visualize(self, save_dir=None, classes=[], show_labels=True, sign_only=False):
        """Visualize labels.

        Args:
            save_dir (str | optional): The path to save visualized images, 
                if it's None then just show the images.
            classes (List[int | str] | optional): To specify the classes to visualize, it's a list contains
                the cls index or class name.
            show_labels (bool): Whether to show label names, default: True.
            sign_only (bool): Only to plot the images with sign, 
                only the invalid bboxes would be plotted in images with sign if labels are not filtered;
                all the bboxes would be plotted in images with sign if labels are filtered;
        """
        if not osp.exists(self.img_dir):
            LOGGER.warning(f"'{self.img_dir}' doesn't exist.")
            return

        import random
        random.shuffle(self.labels)
        classes = [self.class_names.index(c) if isinstance(c, str) else c for c in classes]
        filter = len(classes)

        pbar = tqdm(self.labels, total=len(self.labels))
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        else:
            cv2.namedWindow("p", cv2.WINDOW_NORMAL)

        from ultralytics.utils.plotting import Annotator, colors
        for label in pbar:
            plotted = False
            try:
                sign = label.get("sign", {})
                if sign_only and len(sign) == 0:
                    continue
                filename = label["img_name"]
                image = cv2.imread(osp.join(self.img_dir, filename))
                if image is None:
                    continue
                annotator = Annotator(image, line_width=2)
                cls = label["cls"]
                bbox = label["bbox"]
                if filter:
                    idx = [i for i in range(len(cls)) if cls[i] in classes]
                    cls = [cls[i] for i in idx]
                    bbox = bbox[idx]
                bbox.convert("xyxy")
                for i, c in enumerate(cls):
                    sign_info = ""
                    for k, v in sign.items():
                        if i in v:
                            sign_info += f" {k}"
                    if sign_only and len(sign_info) == 0:
                        continue
                    l = self.class_names[int(c)] + sign_info if show_labels else None
                    annotator.box_label(bbox.data[i], l, color=colors(int(c)))
                    plotted = True
            except Exception as e:
                LOGGER.warning(e)
                continue
            if not plotted:
                continue
            if save_dir:
                cv2.imwrite(osp.join(save_dir, filename), annotator.result())
            else:
                cv2.imshow("p", annotator.result())
                if cv2.waitKey(0) == ord("q"):
                    break

    def get_xml_template(self):
        anntemp = """\
<annotation>
    <folder>VOC</folder>
    <filename>%s</filename>
    <source>
        <database>My Database</database>
        <annotation>COCO</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>company</name>
    </owner>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
%s
</annotation>
        """
        objtemp = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <occluded>0</occluded>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
        """
        return anntemp, objtemp

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

