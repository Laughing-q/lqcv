from abc import ABCMeta, abstractmethod
from lqcv.utils.log import LOGGER
from lqcv.utils.plot import plot_one_box, colors
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from tabulate import tabulate
import os.path as osp
import cv2
import shutil
import json



class BaseConverter(metaclass=ABCMeta):
    def __init__(self, label_dir, class_names=None, img_dir=None) -> None:
        super().__init__()
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
        pbar.desc = "Convert YOLO to COCO: "
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

    def toXML(self):
        pass

    def toYOLO(self):
        pass

    @abstractmethod
    def read_labels(self, label_dir):
        pass

    def visualize(self, save_dir=None):
        if not osp.exists(self.img_dir):
            LOGGER.warning(f"'{self.img_dir}' doesn't exist.")
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

