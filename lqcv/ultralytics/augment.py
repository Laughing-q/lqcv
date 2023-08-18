from ultralytics.data.augment import Mosaic, RandomPerspective
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
        neg_idxs = random.choices(range(len(self.dataset.neg_files)), k=self.n - 1) if num_neg else []
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
        self.area_thr = area_thr

    def box_candidates(self, box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):
        # NOTE: use self.area_thr instead of the original arg `area_thr`.
        return super().box_candidates(box1, box2, wh_thr, ar_thr, self.area_thr, eps)
