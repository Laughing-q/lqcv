from PIL import Image, ImageFile
import numpy as np

# TODO: update this code
ImageFile.LOAD_TRUNCATED_IMAGES = True


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
    old_size = im.size
    ratio = get_raito(new_size, old_size)
    im = im.resize((int(old_size[0] * ratio), int(old_size[1] * ratio)))
    return im


def get_wh(a, b):
    return np.random.randint(a, b)


def paste2(sample1, sample2, background, scale=1.2):
    sample1 = Image.open(sample1)
    d_w1, d_h1 = sample1.size

    sample2 = Image.open(sample2)
    d_w2, d_h2 = sample2.size

    background = Image.open(background)
    background = background.resize((int((d_w1 + d_w2) * scale), int((d_h1 + d_h2) * scale)))
    bw, bh = background.size

    x1, y1 = get_wh(0, int(d_w1 * scale) - d_w1), get_wh(0, bh - d_h1)
    x2, y2 = get_wh(int(d_w1 * scale), bw - d_w2), get_wh(0, bh - d_h2)

    background.paste(sample1, (x1, y1))
    background.paste(sample2, (x2, y2))

    return np.array(background), (x1, y1, x2, y2), background


def paste1(foreground, background, bg_size, fg_scale=1.5):
    foreground = Image.open(foreground)
    background = Image.open(background)
    background = imresize(background, bg_size)
    bw, bh = background.size
    new_w, new_h = int(bw / fg_scale), int(bh / fg_scale)
    foreground = imresize(foreground, (new_w, new_h))

    fw, fh = foreground.size
    x1, y1 = get_wh(0, bw - fw), get_wh(0, bh - fh)
    background.paste(foreground, (x1, y1))

    return np.array(background.convert("RGB"))[:, :, ::-1], (x1, y1), background, (fw, fh)


if __name__ == "__main__":
    import cv2
    output, coord1, _, _ = paste1(
        "/home/laughing/codes/lqcv/lqcv/assets/bus.jpg",
        "/home/laughing/codes/lqcv/lqcv/assets/zidane.jpg",
        bg_size=640,
        fg_scale=2,
    )
    print(coord1)
    cv2.imshow("P", output)
    cv2.waitKey(0)
