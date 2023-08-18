from lqcv.utils.plot import cv2_imshow
import numpy as np
import random
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


def paste1(foreground, background, bg_size=None, fg_scale=1.5):
    """Paste one foreground image to another background image.

    Args:
        foreground (str | np.ndarray): The foreground image.
        background (str | np.ndarray): The background image.
        bg_size (int | tuple): The dst size of background, it should be (w, h) if it's a tuple.
        fg_scale (float): Foreground scale.

    Returns:
        pasted image, pasted position, foreground size.

    """
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
    background[y1 : y1 + fh, x1 : x1 + fw] = foreground

    return background, (x1, y1), (fw, fh)


def paste1_mask(foreground, background, bg_size=None, fg_scale=1.5):
    """Paste one foreground image to another background images, and the foreground one is a binary image.

    Args:
        foreground (str | np.ndarray): The foreground image.
        background (str | np.ndarray): The background image.
        bg_size (int | tuple): The dst size of background, it should be (w, h) if it's a tuple.
        fg_scale (float): Foreground scale.

    Returns:
        pasted image, pasted position, foreground size.

    """
    if isinstance(foreground, str):
        foreground = cv2.imread(foreground)
    if isinstance(background, str):
        background = cv2.imread(background)
    if bg_size is not None:
        background = imresize(background, bg_size)
    bh, bw = background.shape[:2]
    new_w, new_h = int(bw / fg_scale), int(bh / fg_scale)
    foreground = imresize(foreground, (new_w, new_h))

    fh, fw = foreground.shape[:2]
    x1, y1 = get_wh(0, bw - fw), get_wh(0, bh - fh)
    mask = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY) > 5
    background[y1 : y1 + fh, x1 : x1 + fw][mask] = foreground[mask]

    return background, (x1, y1), (fw, fh)


def paste_masks(foregrounds, background, bg_size=None, fg_scale=1.5):
    """Paste multiple foreground images to one background images, and the foreground ones are binary images.

    Args:
        foreground (List[str | np.ndarray]): The foreground images.
        background (str | np.ndarray): The background image.
        bg_size (int | tuple): The dst size of background, it should be (w, h) if it's a tuple.
        fg_scale (float): Foreground scale.

    Returns:
        pasted image, pasted coordinates.

    """
    # scales for (w, h)
    scales = {
        1: [1, 1],
        2: [1, 2],
        3: [2, 2],
        4: [2, 2],
        5: [2, 3],
        6: [2, 3],
    }
    foregrounds = [cv2.imread(f) if isinstance(f, str) else f for f in foregrounds]
    num_fg = len(foregrounds)
    assert 0 < num_fg <= 6

    if isinstance(background, str):
        background = cv2.imread(background)

    scale = scales[num_fg]
    scale = scale if np.random.uniform() < 0.5 else list(reversed(scale))
    if bg_size is not None:
        background = imresize(background, bg_size)
    bh, bw = background.shape[:2]
    x_patches, y_patches = np.meshgrid(
        np.linspace(0, bw, scale[0] + 1),
        np.linspace(0, bh, scale[1] + 1)
    )
    patchs = np.stack([x_patches, y_patches], axis=-1)
    ranges = [(patchs[i][j], patchs[i + 1][j + 1]) for i in range(scale[1]) for j in range(scale[0])]
    random.shuffle(ranges)
    ltwh = []
    for i, (start, end) in enumerate(ranges):
        if i >= num_fg:
            break
        pw, ph = int(end[0] - start[0]), int(end[1] - start[1])
        new_w, new_h = int(pw / fg_scale), int(ph / fg_scale)
        foreground = imresize(foregrounds[i], (new_w, new_h))
        fh, fw = foreground.shape[:2]

        x = get_wh(int(start[0]), int(end[0]) - fw)
        y = get_wh(int(start[1]), int(end[1]) - fh)
        mask = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY) > 5
        background[y : y + fh, x : x + fw][mask] = foreground[mask]
        ltwh.append([x, y, fw, fh])

    return background, ltwh


if __name__ == "__main__":
    import cv2

    # for i in range(5):
    #     output, coord1, _ = paste1(
    #         "/home/laughing/codes/lqcv/lqcv/assets/bus.jpg",
    #         "/home/laughing/codes/lqcv/lqcv/assets/zidane.jpg",
    #         bg_size=640,
    #         fg_scale=np.random.uniform(1.5, 3),
    #     )
    #     print(coord1)
    #     cv2.imshow("P", output)
    #     cv2.waitKey(0)

    # for i in range(5):
    #     output, coord1, _ = paste1_mask(
    #         "/d/dataset/smoke_fire/smoke_fire/total/masks/fire/(2)_0.jpg",
    #         "/home/laughing/codes/lqcv/lqcv/assets/zidane.jpg",
    #         bg_size=640,
    #         fg_scale=np.random.uniform(1.5, 3),
    #     )
    #     print(coord1)
    #     cv2.imshow("P", output)
    #     cv2.waitKey(0)

    for i in range(5):
        output, _ = paste_masks(
            [
                "/d/dataset/smoke_fire/smoke_fire/total/masks/smoke/(71)_1.jpg",
                "/d/dataset/smoke_fire/smoke_fire/total/masks/fire/(2)_0.jpg",
                "/d/dataset/smoke_fire/smoke_fire/total/masks/fire/(2)_0.jpg",
             ],
            "/home/laughing/codes/lqcv/lqcv/assets/zidane.jpg",
            # bg_size=640,
            fg_scale=np.random.uniform(1.5, 3),
        )
        cv2_imshow(output)
