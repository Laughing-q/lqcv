# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Plotting utils
"""


import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from itertools import repeat

from .color import colors


class Annotator:
    def __init__(
        self,
        im,
        line_width=None,
        font_size=None,
        font=None,
    ):
        assert (
            im.data.contiguous
        ), "Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images."
        self.pil = font is not None
        if self.pil:  # use PIL
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            self.font = ImageFont.truetype(
                str(font),
                size=font_size or max(round(sum(self.im.size) / 2 * 0.035), 12),
            )
        else:  # use cv2
            self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width

    def box_label(
        self, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255)
    ):
        # Add one xyxy box to image with label
        if self.pil:
            self.draw.rectangle(box, width=self.lw, outline=color)  # box
            if label:
                w, h = self.font.getsize(label)  # text width, height
                outside = box[1] - h >= 0  # label fits outside box
                self.draw.rectangle(
                    [
                        box[0],
                        box[1] - h if outside else box[1],
                        box[0] + w + 1,
                        box[1] + 1 if outside else box[1] + h + 1,
                    ],
                    fill=color,
                )
                # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
                self.draw.text(
                    (box[0], box[1] - h if outside else box[1]),
                    label,
                    fill=txt_color,
                    font=self.font,
                )
        else:  # cv2
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(
                self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA
            )
            if label:
                tf = max(self.lw - 1, 1)  # font thickness
                w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[
                    0
                ]  # text width, height
                outside = p1[1] - h - 3 >= 0  # label fits outside box
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(
                    self.im,
                    label,
                    (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    self.lw / 3,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA,
                )

    def rectangle(self, xy, fill=None, outline=None, width=1):
        # Add rectangle to image (PIL-only)
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255)):
        # Add text to image (PIL-only)
        _, h = self.font.getsize(text)  # text width, height
        self.draw.text((xy[0], xy[1] - h + 1), text, fill=txt_color, font=self.font)

    def result(self):
        # Return annotated image as array
        return np.asarray(self.im)


class Visualizer(object):
    """Visualization of one model."""

    def __init__(self, names) -> None:
        super().__init__()
        self.names = names

    def draw_one_img(self, img, output, vis_conf=0.4):
        """Visualize one images.

        Args:
            imgs (numpy.ndarray): one image.
            outputs (torch.Tensor): one output, (num_boxes, classes+5)
            vis_confs (float, optional): Visualize threshold.
        Return:
            img (numpy.ndarray): Image after visualization.
        """
        if isinstance(output, list):
            output = output[0]
        if output is None or len(output) == 0:
            return img
        for (*xyxy, conf, cls) in reversed(output[:, :6]):
            if conf < vis_conf:
                continue
            label = "%s %.2f" % (self.names[int(cls)], conf)
            color = colors(int(cls))
            plot_one_box(xyxy, img, label=label, color=color, line_thickness=2)
        return img

    def draw_multi_img(self, imgs, outputs, vis_confs=0.4):
        """Visualize multi images.

        Args:
            imgs (List[numpy.array]): multi images.
            outputs (List[torch.Tensor]): multi outputs, List[num_boxes, classes+5].
            vis_confs (float | tuple[float], optional): Visualize threshold.
        Return:
            imgs (List[numpy.ndarray]): Images after visualization.
        """
        if isinstance(vis_confs, float):
            vis_confs = list(repeat(vis_confs, len(imgs)))
        assert len(imgs) == len(outputs) == len(vis_confs)
        for i, output in enumerate(outputs):  # detections per image
            self.draw_one_img(imgs[i], output, vis_confs[i])
        return imgs

    def draw_imgs(self, imgs, outputs, vis_confs=0.4):
        if isinstance(imgs, np.ndarray):
            return self.draw_one_img(imgs, outputs, vis_confs)
        else:
            return self.draw_multi_img(imgs, outputs, vis_confs)

    def __call__(self, imgs, outputs, vis_confs=0.4):
        return self.draw_imgs(imgs, outputs, vis_confs)



def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    import random

    # Plots one bounding box on image img
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def plot_masks(img, masks, colors, alpha=0.5):
    """
    Args:
        img (tensor): img on cuda, shape: [3, h, w], range: [0, 1]
        masks (tensor): predicted masks on cuda, shape: [n, h, w]
        colors (List[List[Int]]): colors for predicted masks, [[r, g, b] * n]
    Return:
        img after draw masks, shape: [h, w, 3]

    transform colors and send img_gpu to cpu for the most time.
    """
    img_gpu = img.clone()
    num_masks = len(masks)
    # [n, 1, 1, 3]
    # faster this way to transform colors
    colors = torch.tensor(colors, device=img.device).float() / 255.0
    colors = colors[:, None, None, :]
    # [n, h, w, 1]
    masks = masks[:, :, :, None]
    masks_color = masks.repeat(1, 1, 1, 3) * colors * alpha
    inv_alph_masks = masks * (-alpha) + 1
    masks_color_summand = masks_color[0]
    if num_masks > 1:
        inv_alph_cumul = inv_alph_masks[: (num_masks - 1)].cumprod(dim=0)
        masks_color_cumul = masks_color[1:] * inv_alph_cumul
        masks_color_summand += masks_color_cumul.sum(dim=0)

    # print(inv_alph_masks.prod(dim=0).shape) # [h, w, 1]
    img_gpu = img_gpu.flip(dims=[0])  # filp channel for opencv
    img_gpu = img_gpu.permute(1, 2, 0).contiguous()
    # [h, w, 3]
    img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand
    return (img_gpu * 255).byte().cpu().numpy()
