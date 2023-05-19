from pathlib import Path
import numpy as np
import cv2
import os.path as osp
import os

IMG_FORMATS = [
    "bmp",
    "jpg",
    "jpeg",
    "png",
    "tif",
    "tiff",
    "dng",
    "webp",
    "mpo",
]  # acceptable image suffixes

VID_FORMATS = [
    "mov",
    "avi",
    "mp4",
    "mpg",
    "mpeg",
    "m4v",
    "wmv",
    "mkv",
    "vdo",
    "flv",
    "ts",
]  # acceptable video suffixes

def verify_image_label(args):
    # Verify one image-label pair
    img_name, imgs_dir, labels_dir, num_classes = args
    im_file = osp.join(imgs_dir, img_name) if imgs_dir is not None else None
    lb_file = osp.join(labels_dir, str(Path(img_name).with_suffix('.txt')))
    nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, corrupt
    try:
        # verify images
        if im_file is not None:
            im = cv2.imread(im_file)
            shape = im.shape[:2]  # h, w
            shape.append(3)    # add channel
            suffix = Path(im_file).suffix[1:]
            assert suffix.lower() in IMG_FORMATS, f'invalid image format {suffix}'
        else:
            shape = None
            img_name = None

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file, 'r') as f:
                l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                l = np.array(l, dtype=np.float32)
            if len(l):
                assert l.shape[1] == 5, f'labels require 5 columns each: {lb_file}'
                assert (l >= 0).all(), f'negative labels: {lb_file}'
                assert (l[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinate labels: {lb_file}'
                assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                assert (l[:, 0] < num_classes).all(), f'label cls index out of range, {l[:, 0]}, {lb_file}'
            else:
                ne = 1  # label empty
                l = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            l = np.zeros((0, 5), dtype=np.float32)
        return img_name, l, shape, nm, nf, ne, nc, ''
    except Exception as e:
        nc = 1
        msg = f'WARNING: Ignoring corrupted image and/or label {im_file}: {e}'
        return [None, None, None, nm, nf, ne, nc, msg]
