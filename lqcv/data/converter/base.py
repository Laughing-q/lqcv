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
    def __init__(self, label_dir, class_names=None, img_dir=None, chunk_size=None) -> None:
        super().__init__()
        assert osp.exists(label_dir), f"The directory/file '{label_dir}' does not exist."

        self.label_dir = label_dir  # adding this attribute since it'd be useful for some scenarios
        self.labels = list()
        self.catCount = defaultdict(int)
        self.catImgCnt = dict()
        self.img_dir = img_dir
        self.class_names = class_names

        self.read_labels(label_dir, chunk_size)

    def toCOCO(self, save_file, classes=None, im_dir=None):
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
        if copy_im:
            os.makedirs(im_dir, exist_ok=True)

        pbar = tqdm(enumerate(self.labels), total=len(self.labels))
        pbar.desc = f"Convert {self.format.upper()} to COCO: "
        for idx, label in pbar:
            h, w = label["shape"][:2]
            image, sub_annotations = dict(), []
            image["file_name"] = label["img_name"]

            image["height"] = h
            image["width"] = w
            image["id"] = idx

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
        cocoDict["categories"] = [
            {"supercategory": "none", "id": c, "name": class_name[c]} for c in range(len(class_name))
        ]
        cocoDict["type"] = "instances"

        # print attrDict
        Path(save_file).parent.mkdir(parents=True, exist_ok=True)
        jsonString = json.dumps(cocoDict, indent=2)
        with open(save_file, "w") as f:
            f.write(jsonString)

        LOGGER.info(f"Convert results: {len(images)}/{len(self.labels)}")

    def toXML(self, save_dir, classes=None, im_dir=None):
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
        if copy_im:
            os.makedirs(im_dir, exist_ok=True)

        anno_temp, obj_temp = self.get_xml_template()
        pbar = tqdm(self.labels, total=len(self.labels))
        pbar.desc = f"Convert {self.format.upper()} to XML: "
        for label in pbar:
            h, w, c = label["shape"]
            cls, bboxes = label["cls"], label["bbox"]
            bboxes.convert("xyxy")
            filename = label["img_name"]

            obj_str = ""
            xml_name = str(Path(filename).with_suffix(".xml"))
            xml_path = osp.join(save_dir, xml_name)
            for i, c in enumerate(cls):
                name = self.class_names[int(c)]

                if name not in class_name:
                    LOGGER.info(f"`{name}` not in {class_name}, ignore")
                    continue

                bbox = bboxes[i].data.squeeze().tolist()
                obj_str += obj_temp % (name, *bbox)
            if len(obj_str):
                f_xml = open(xml_path, "w")
                f_xml.write(anno_temp % (filename, w, h, c, obj_str))
                f_xml.close()
                if copy_im:
                    shutil.copy(osp.join(self.img_dir, label["img_name"]), im_dir)

        LOGGER.info(f"Convert results: {len(os.listdir(save_dir))}/{len(self.labels)}")

    def toYOLO(self, save_dir, classes=None, classes_idx=None, im_dir=None, single_cls=False):
        """Convert labels to yolo format.

        Args:
            save_dir (str): Save dir for the dst txt files.
            classes (Optional | List[str]): Filter the class if given.
            classes_idx (Optional | List[int]): Update the class idx if given.
            im_dir (Optional | str): Move the images to im_dir if given and `classes` is also given.
            single_cls (Optional | bool): Whether to treat all the classes to one class, default: False.
        """
        if self.format == "yolo" and classes is None and classes_idx is None:
            LOGGER.info(
                "Current format is YOLO! there's no need to convert it since `classes` and `classes_idx` are not given."
            )
            return
        assert not (classes is not None and classes_idx is not None), (
            "`classes` and `classes_idx` are mutually exclusive."
        )
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

            label_name = str(Path(filename).with_suffix(".txt"))
            label_path = osp.join(save_dir, label_name)

            label = ""
            for i, c in enumerate(cls):
                name = self.class_names[int(c)]

                if name not in class_name:
                    LOGGER.info(f"`{name}` not in {class_name}, ignore")
                    continue

                cx, cy, bw, bh = bboxes[i].data.squeeze().tolist()
                category_id = 0 if single_cls else class_name.index(name)
                if classes_idx is not None:
                    category_id = classes_idx[category_id]
                cx /= w
                cy /= h
                bw /= w
                bh /= h

                label += "%s %s %s %s %s\n" % (category_id, cx, cy, bw, bh)

            if len(label):
                f_label = open(label_path, "w")
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
        self.check_results = {"pixel": defaultdict(int), "iou": defaultdict(int)}
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
                pixel_pick = (bboxes.data[:, 2:] >= min_pixel).all(1)
            iou = np.triu(Boxes.iou(bboxes, bboxes), k=1)
            iou_idx = np.unique(np.concatenate(np.nonzero((iou > iou_thres))))
            if len(iou_idx):
                LOGGER.warning(f"WARNING ⚠️{filename} got boxes with overlaps greater than {iou_thres}!")
                # NOTE: no need to keep indices if filter=True, as the indices would be incorrect anyway.
                label["sign"]["overlap"] = iou_idx if not filter else list(range(len(bboxes)))
                iou_pick = iou.max(axis=0) <= iou_thres

                paired_idx = np.stack(np.nonzero(iou > iou_thres), axis=1)
                cls = np.array(label["cls"], dtype=np.int32)
                for idx in paired_idx:
                    n1, n2 = (self.class_names[i] for i in cls[idx])
                    if (n2, n1) in self.check_results["iou"]:
                        self.check_results["iou"][(n2, n1)] += 1
                    else:
                        self.check_results["iou"][(n1, n2)] += 1
            if filter and (len(iou_idx) or len(pixel_idx)):
                ori_len = len(bboxes)
                pick = iou_pick & pixel_pick
                label["bbox"] = bboxes[pick]
                label["cls"] = [c for i, c in enumerate(label["cls"]) if pick[i]]
                LOGGER.info(f"Filter results: {len(label['bbox'])}/{ori_len}")

    @abstractmethod
    def read_labels(self, label_dir, chunk_size):
        pass

    def visualize(self, save_dir=None, classes=[], show_labels=True, sign_only=False, shuffle=True, im_names=None):
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
            shuffle (bool): Whether to shuffle the labels, this is for the use case of
                randomly check images sometimes.
            im_names (Optional | List[str]): Only visualize specific images if its image names provided.
        """
        if not osp.exists(self.img_dir):
            LOGGER.warning(f"'{self.img_dir}' doesn't exist.")
            return

        if shuffle:
            import random

            random.shuffle(self.labels)
        classes = [self.class_names.index(c) if isinstance(c, str) else c for c in classes]
        filter = len(classes)

        pbar = tqdm(self.labels, total=len(self.labels))
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        else:
            cv2.namedWindow("p", cv2.WINDOW_NORMAL)
        if im_names is not None and isinstance(im_names, str):
            im_names = [im_names]

        from ultralytics.utils.plotting import Annotator, colors

        for label in pbar:
            plotted = False
            try:
                sign = label.get("sign", {})
                if sign_only and len(sign) == 0:
                    continue
                filename = label["img_name"]
                if im_names is not None and filename not in im_names:
                    continue
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
                    l = f"{self.class_names[int(c)]}" + sign_info if show_labels else None
                    annotator.box_label(bbox.data[i], l, color=colors(int(c), True))
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

    def move_empty(self, save_dir, lb_suffix=".txt"):
        """Move empty labels among with corresponding images to a new folder.

        Args:
            save_dir (str): The dst save folder.
            lb_suffix (str): The suffix of the label file, could be ".txt" or ".xml".
        """
        # NOTE: It seems this function is not needed since empty labels won't be kept in self.labels
        assert self.img_dir is not None
        assert lb_suffix in {".txt", ".xml"}
        img_dir, label_dir = Path(self.img_dir), Path(self.label_dir)
        save_dir = Path(save_dir)
        save_im_dir, save_lb_dir = save_dir / "images", save_dir / "xmls"
        save_im_dir.mkdir(parents=True, exist_ok=True)
        save_lb_dir.mkdir(parents=True, exist_ok=True)
        for label in tqdm(self.labels, desc="Moving empty labels and images"):
            if len(label["cls"]) > 0:
                continue
            filename = label["img_name"]
            im_file = img_dir / filename
            lb_file = label_dir / Path(filename).with_suffix(lb_suffix)
            if not im_file.exists():
                print(f"{im_file} does not exist!")
            if not lb_file.exists():
                print(f"{lb_file} does not exist!")
            shutil.move(im_file, save_im_dir)
            shutil.move(lb_file, save_lb_dir)

    @staticmethod
    def get_xml_template():
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
        """The string representation of the converter, which includes the format, class names and label info."""
        return self._get_label_info()

    def _get_label_info(self, sort_by_count=False):
        """Get the label information in a table format.

        Args:
            sort_by_count (bool): Whether to sort the table by image count, default: False.

        Returns:
            (str): The label information in a table format.
        """
        total_img, total_obj = 0, 0
        cat_table = []
        for i, c in enumerate(self.class_names):
            if c in self.catImgCnt and c in self.catCount:
                total_img += self.catImgCnt[c]
                total_obj += self.catCount[c]
                cat_table.append((str(i), str(c), self.catImgCnt[c], self.catCount[c]))
            else:
                cat_table.append((str(i), str(c), 0, 0))

        if sort_by_count:
            cat_table.sort(key=lambda x: x[2], reverse=True)
        cat_table += [(" ", "total", total_img, total_obj)]
        return "\n" + tabulate(
            cat_table,
            headers=["Id", "Category", "ImageCnt", "ClassCnt"],
            tablefmt="fancy_grid",
            missingval="None",
        )

    def updateCntInfo(self):
        """
        Update the count information.

        Use this whenever the labels are updated.
        """
        catCount = np.zeros(len(self.class_names), dtype=np.int32)
        catImgCnt = np.zeros(len(self.class_names), dtype=np.int32)
        for l in tqdm(self.labels, desc="Updating count info"):
            cls = l["cls"]
            if len(cls) == 0:
                continue
            catImgCnt[cls] += 1
            catCount += np.bincount(np.array(cls), minlength=len(self.class_names))
        for i, name in enumerate(self.class_names):
            self.catCount[name] = catCount[i]
            self.catImgCnt[name] = catImgCnt[i]

    def to_dict(self):
        """Converts the labels to a dictionary, it's useful to use dict format for comparison/matching between multiple group of labels.

        Iterates over the labels and creates a dictionary where the keys are
        image names and the values are the corresponding label data.

        Returns:
            dict: A dictionary with image names as keys and label data as values.
        """
        labels = {}
        pbar = tqdm(self.labels, total=len(self.labels))
        for label in pbar:
            labels[label["img_name"]] = label
        return labels

    def get_statistics(self):
        """Visualize dataset statistics with histograms for box sizes and category distribution.

        Args:
            save_path (str | optional): Path to save the figure. If None, displays the plot.
        """
        import matplotlib.pyplot as plt

        plt.switch_backend("Agg")

        assert len(self.labels), "No labels detected!"

        # Collect box size statistics
        area_ranges = {
            "Small(0-32²)": (0, 32**2),
            "Medium(32²-96²)": (32**2, 96**2),
            "Large(>96²)": (96**2, float("inf")),
        }
        box_counts = {"Small(0-32²)": 0, "Medium(32²-96²)": 0, "Large(>96²)": 0}

        for label in tqdm(self.labels, desc="Analyzing box sizes"):
            areas = label["bbox"].areas
            box_counts["Small(0-32²)"] += (
                (areas >= area_ranges["Small(0-32²)"][0]) & (areas < area_ranges["Small(0-32²)"][1])
            ).sum()
            box_counts["Medium(32²-96²)"] += (
                (areas >= area_ranges["Medium(32²-96²)"][0]) & (areas < area_ranges["Medium(32²-96²)"][1])
            ).sum()
            box_counts["Large(>96²)"] += (
                (areas >= area_ranges["Large(>96²)"][0]) & (areas < area_ranges["Large(>96²)"][1])
            ).sum()

        # Collect category statistics
        cat_names = []
        cat_counts = []
        for name in self.class_names:
            if name in self.catCount and self.catCount[name] > 0:
                cat_names.append(name)
                cat_counts.append(self.catCount[name])

        # Create figure with two subplots
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Box size distribution
        sizes = list(box_counts.keys())
        counts = list(box_counts.values())
        colors_size = ["#3498db", "#2ecc71", "#e74c3c"]
        bars1 = ax1.bar(sizes, counts, color=colors_size, edgecolor="black", linewidth=1.2)
        ax1.set_xlabel("Box Size Category", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Count", fontsize=12, fontweight="bold")
        ax1.set_title("Box Size Distribution", fontsize=14, fontweight="bold")
        ax1.grid(axis="y", alpha=0.3, linestyle="--")

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        # Plot 2: Category distribution
        colors_cat = plt.cm.tab20(np.linspace(0, 1, len(cat_names)))
        bars2 = ax2.bar(range(len(cat_names)), cat_counts, color=colors_cat, edgecolor="black", linewidth=1.2)
        ax2.set_xlabel("Category", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Count", fontsize=12, fontweight="bold")
        ax2.set_title("Category Distribution", fontsize=14, fontweight="bold")
        ax2.set_xticks(range(len(cat_names)))
        ax2.set_xticklabels(cat_names, rotation=45, ha="right")
        ax2.grid(axis="y", alpha=0.3, linestyle="--")

        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        plt.tight_layout()

        save_path = "dataset_statistics.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        LOGGER.info(f"Statistics visualization saved to {save_path}")
        plt.close()
        LOGGER.info(self._get_label_info(sort_by_count=True))
        LOGGER.info(
            f"Small box cnt: {box_counts['Small(0-32²)']}\nMedium box cnt: {box_counts['Medium(32²-96²)']}\nLarge box cnt: {box_counts['Large(>96²)']}"
        )
