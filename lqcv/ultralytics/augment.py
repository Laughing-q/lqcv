from ultralytics.data.augment import Mosaic, RandomPerspective
from ultralytics.utils.instance import Instances
import numpy as np
import random
import cv2


class NMosaic(Mosaic):
    """Mosaic that can concatenate negtive(N) images images."""

    def __init__(self, dataset, imgsz=640, p=1, n=4):
        super().__init__(dataset, imgsz, p, n)

    def __call__(self, labels):
        """Applies pre-processing transforms and mixup/mosaic transforms to labels data."""
        if random.uniform(0, 1) > self.p:
            ln = len(self.dataset.neg_files)
            # NOTE: there's 0.2 rate to load negative image when mosaic is not used
            if ln and random.uniform(0, 1) > 0.8:
                return self.dataset.get_neg_image(random.choice(range(ln)))
            return labels

        # Get index of one or three other images
        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]

        # Get images information will be used for Mosaic
        num_neg = random.randint(0, 2) if len(self.dataset.neg_files) else 0
        neg_idxs = (
            random.choices(range(len(self.dataset.neg_files)), k=self.n - 1) if num_neg else []
        )
        patch_idxs = random.choices(range(self.n - 1), k=num_neg) if num_neg else []

        mix_labels = []
        for i, idx in enumerate(indexes):
            label = (
                self.dataset.get_neg_image(neg_idxs[i])
                if i in patch_idxs
                else self.dataset.get_image_and_label(idx)
            )
            mix_labels.append(label)

        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels):
                mix_labels[i] = self.pre_transform(data)
        labels["mix_labels"] = mix_labels

        # Mosaic or MixUp
        labels = self._mix_transform(labels)
        labels.pop("mix_labels", None)
        return labels


class Resize:
    def __init__(self, size=640):
        """Converts an image from numpy array to PyTorch tensor."""
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size

    def __call__(self, im):  # im = np.array HWC
        return cv2.resize(im, (self.w, self.h), interpolation=cv2.INTER_LINEAR)


class ARandomPerspective(RandomPerspective):
    """ultralytics's RandomPerspective but with an extra arg `area_thr`."""

    def __init__(
        self,
        degrees=0,
        translate=0.1,
        scale=0.5,
        shear=0,
        perspective=0,
        border=(0, 0),
        pre_transform=None,
        area_thr=0.1,
    ):
        super().__init__(degrees, translate, scale, shear, perspective, border, pre_transform)
        self.area_thr = area_thr   # list

    def box_candidates(self, box1, box2, cls, wh_thr=2, ar_thr=100, eps=1e-16):
        """
        Compute box candidates based on a set of thresholds. This method compares the characteristics of the boxes
        before and after augmentation to decide whether a box is a candidate for further processing.

        Args:
            box1 (numpy.ndarray): The 4,n bounding box before augmentation, represented as [x1, y1, x2, y2].
            box2 (numpy.ndarray): The 4,n bounding box after augmentation, represented as [x1, y1, x2, y2].
            cls (numpy.ndarray): The class idx of gt boxes.
            wh_thr (float, optional): The width and height threshold in pixels. Default is 2.
            ar_thr (float, optional): The aspect ratio threshold. Default is 100.
            eps (float, optional): A small epsilon value to prevent division by zero. Default is 1e-16.

        Returns:
            (numpy.ndarray): A boolean array indicating which boxes are candidates based on the given thresholds.
        """
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        area_thr = np.array(self.area_thr)[cls.astype(np.int)] if isinstance(self.area_thr, list) else self.area_thr
        if isinstance(area_thr, list) and len(area_thr) == 1:
            area_thr = area_thr[0]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
        return (
            (w2 > wh_thr)
            & (h2 > wh_thr)
            & (w2 * h2 / (w1 * h1 + eps) > area_thr)
            & (ar < ar_thr)
        )  # candidates

    def __call__(self, labels):
        """
        Affine images and targets.

        Args:
            labels (dict): a dict of `bboxes`, `segments`, `keypoints`.
        """
        if self.pre_transform and "mosaic_border" not in labels:
            labels = self.pre_transform(labels)
        labels.pop("ratio_pad", None)  # do not need ratio pad

        img = labels["img"]
        cls = labels["cls"]
        instances = labels.pop("instances")
        # Make sure the coord formats are right
        instances.convert_bbox(format="xyxy")
        instances.denormalize(*img.shape[:2][::-1])

        border = labels.pop("mosaic_border", self.border)
        self.size = img.shape[1] + border[1] * 2, img.shape[0] + border[0] * 2  # w, h
        # M is affine matrix
        # Scale for func:`box_candidates`
        img, M, scale = self.affine_transform(img, border)

        bboxes = self.apply_bboxes(instances.bboxes, M)

        segments = instances.segments
        keypoints = instances.keypoints
        # Update bboxes if there are segments.
        if len(segments):
            bboxes, segments = self.apply_segments(segments, M)

        if keypoints is not None:
            keypoints = self.apply_keypoints(keypoints, M)
        new_instances = Instances(bboxes, segments, keypoints, bbox_format="xyxy", normalized=False)
        # Clip
        new_instances.clip(*self.size)

        # Filter instances
        instances.scale(scale_w=scale, scale_h=scale, bbox_only=True)
        # Make the bboxes have the same scale with new_bboxes
        i = self.box_candidates(box1=instances.bboxes.T, box2=new_instances.bboxes.T, cls=cls)
        labels["instances"] = new_instances[i]
        labels["cls"] = cls[i]
        labels["img"] = img
        labels["resized_shape"] = img.shape[:2]
        return labels
