import cv2
import numpy as np

def pHash(img):
    img = cv2.resize(img, (32, 32))  # , interpolation=cv2.INTER_CUBIC

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(gray))
    dct_roi = dct[0:8, 0:8].flatten()
    avreage = np.mean(dct_roi)
    hash = np.where(dct_roi > avreage, 1, 0).flatten()
    return list(hash)


def img_hash(img):
    """create this function to directly compare the pixel difference in 8x8 size"""
    img = cv2.resize(img, (8, 8))  # , interpolation=cv2.INTER_CUBIC
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return list(gray.flatten())


def cmpHash_np(hash1, hash2):
    """
    :param hash1: (N, 64), (64, )
    :param hash2: (M, 64)
    :return:
    """
    x = (hash1[None, :] != hash2[:, None]).sum(2, dtype=np.uint8)
    return x


def cmpImgHash_np(hash1, hash2):
    """
    :param hash1: (N, 64), (64, )
    :param hash2: (M, 64)
    :return:
    """
    x = np.abs(hash1[None, :] - hash2[:, None]).sum(2).astype(np.float32)
    x = x / 255.
    return x
