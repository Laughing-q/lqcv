import numpy as np
import cv2


def get_raito(new_size, original_size):
    """Get the ratio bewtten input_size and original_size"""
    # # mmdet way
    # iw, ih = new_size
    # ow, oh = original_size
    # max_long_edge = max(iw, ih)
    # max_short_edge = min(iw, ih)
    # ratio = min(max_long_edge / max(ow, oh), max_short_edge / min(ow, oh))
    # return ratio

    # yolov5 way
    return min(new_size[0] / original_size[0], new_size[1] / original_size[1])


def imresize(im, new_size):
    """Resize the img with new_size by PIL(keep aspect).

    Args:
        img (PIL): The original image.
        new_size (tuple): The new size(w, h).
    """
    if isinstance(new_size, int):
        new_size = (new_size, new_size)
    h, w = im.shape[:2]
    ratio = get_raito(new_size, (w, h))
    im = cv2.resize(im, (int(w * ratio), int(h * ratio)))
    return im


def get_wh(a, b):
    return np.random.randint(a, b)


def paste1(foreground, background, bg_size, fg_scale=1.5):
    if isinstance(foreground, str):
        foreground = cv2.imread(foreground)
    if isinstance(background, str):
        background = cv2.imread(background)
    background = imresize(background, bg_size)
    bh, bw = background.shape[:2]
    new_w, new_h = int(bw / fg_scale), int(bh / fg_scale)
    foreground = imresize(foreground, (new_w, new_h))

    fh, fw = foreground.shape[:2]
    x1, y1 = get_wh(0, bw - fw), get_wh(0, bh - fh)
    background[y1:y1+fh, x1:x1+fw] = foreground

    return background, (x1, y1), (fw, fh)


if __name__ == "__main__":
    import cv2

    for i in range(5):
        output, coord1, _ = paste1(
            "/home/laughing/codes/lqcv/lqcv/assets/bus.jpg",
            "/home/laughing/codes/lqcv/lqcv/assets/zidane.jpg",
            bg_size=640,
            fg_scale=np.random.uniform(1.5, 3),
        )
        print(coord1)
        cv2.imshow("P", output)
        cv2.waitKey(0)
