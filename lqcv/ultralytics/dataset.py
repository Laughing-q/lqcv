from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import Compose
from ultralytics.utils.instance import Instances
from ultralytics.utils import LOGGER, colorstr
from .augment import NBMosaic
import numpy as np
import math
import cv2
import os

class NBYOLODataset(YOLODataset):
    def __init__(self, *args, data=None, use_segments=False, use_keypoints=False, **kwargs):
        super().__init__(*args, data=data, use_segments=use_segments, use_keypoints=use_keypoints, **kwargs)
        self.neg_files, self.bg_files = self.get_neg_and_bg(os.getenv("NEG_DIR", ""), 
                                                            os.getenv("BG_DIR", ""))

    def get_neg_and_bg(self, neg_dir, bg_dir):
        """Get negative pictures and background pictures."""
        img_neg_files, img_bg_files = [], []
        if os.path.isdir(neg_dir):
            img_neg_files = [os.path.join(neg_dir, i) for i in os.listdir(neg_dir)]
            LOGGER.info(
                colorstr("Negative dir: ")
                + f"'{neg_dir}', using {len(img_neg_files)} pictures from the dir as negative samples during training"
            )

        if os.path.isdir(bg_dir):
            img_bg_files = [os.path.join(bg_dir, i) for i in os.listdir(bg_dir)]
            LOGGER.info(
                colorstr("Background dir: ")
                + f"{bg_dir}, using {len(img_bg_files)} pictures from the dir as background during training"
            )
        return img_neg_files, img_bg_files

    def load_other_image(self, im_file):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im = cv2.imread(im_file)  # BGR
        if im is None:
            raise FileNotFoundError(f'Image Not Found {im_file}')
        h0, w0 = im.shape[:2]  # orig hw
        r = self.imgsz / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
            im = cv2.resize(im, (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz)),
                            interpolation=interp)
        return im, (h0, w0), im.shape[:2]

    def get_neg_image(self, index):
        """Get negative image."""
        label = {}
        label['img'], label['ori_shape'], label['resized_shape'] = self.load_other_image(self.neg_files[index])
        label['ratio_pad'] = (label['resized_shape'][0] / label['ori_shape'][0],
                              label['resized_shape'][1] / label['ori_shape'][1])  # for evaluation
        label['cls'] = np.zeros((0, 1), dtype=np.float32)
        return self.update_labels_info(label, neg=True)

    def update_labels_info(self, label, neg=False):
        if neg:
            # Return empty Instances if neg
            nkpt, ndim = self.data.get('kpt_shape', (0, 0))
            keypoints = np.zeros((0, nkpt, ndim), dtype=np.float32) if self.use_keypoints else None
            label['instances'] = Instances(bboxes=np.zeros((0, 4), dtype=np.float32), keypoints=keypoints)
            return label
        bboxes = label.pop('bboxes')
        segments = label.pop('segments')
        keypoints = label.pop('keypoints', None)
        bbox_format = label.pop('bbox_format')
        normalized = label.pop('normalized')
        label['instances'] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    def build_transforms(self, hyp=None):
        ori_trans = super().build_transforms(hyp)
        if isinstance(ori_trans.transforms[0], Compose):
            ori_trans.transforms[0].transforms[0] = NBMosaic(self, imgsz=self.imgsz, p=hyp.mosaic)
        return ori_trans
