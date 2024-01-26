import torch
import numpy as np


def stack(arrays, dim=0):
    """stack arrays for both torch and numpy types."""
    assert isinstance(arrays, (list, tuple))
    return torch.stack(arrays, dim=dim) if isinstance(arrays[0], torch.Tensor) else np.stack(arrays, axis=dim)


def cat(arrays, dim=0):
    """cat arrays for both torch and numpy types."""
    assert isinstance(arrays, (list, tuple))
    return torch.cat(arrays, dim=dim) if isinstance(arrays[0], torch.Tensor) else np.concatenate(arrays, axis=dim)


def clip_coords(boxes, shape):
    """
    Args:
        boxes (torch.Tensor | np.ndarray): xyxy format, (N, 4).
        shape (tuple): (height, width).
    """
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xywh2ltwh(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, w, h] where xy1=top-left
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    return y


def xyxy2ltwh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x1, y1, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def ltwh2xywh(x):
    # Convert nx4 boxes from [x1, y1, w, h] to [x, y, w, h] where xy1=top-left, xy=center
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] + x[:, 2] / 2  # center x
    y[:, 1] = x[:, 1] + x[:, 3] / 2  # center y
    return y


def ltwh2xyxy(x):
    # Convert nx4 boxes from [x1, y1, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 2] = x[:, 2] + x[:, 0]  # width
    y[:, 3] = x[:, 3] + x[:, 1]  # height
    return y


def nms_numpy(boxes, scores, cls, threshold, method=None, agnostic=False):
    """NMS for numpy.

    Args:
        boxes (np.ndarray): numpy(N, 4), xyxy.
        scores (np.ndarray): numpy(N, ).
        cls (np.ndarray): numpy(N, )
        threshold (np.ndarray): threshold.
        method (str): filter method.
        agnostic (bool): agnostic-nms or not.

    Returns:

    """
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int8)
    max_wh = 4096
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if isinstance(cls, torch.Tensor):
        cls = cls.cpu().numpy()

    if boxes.ndim == 1:
        boxes = boxes[None, :]
    assert boxes.shape[1] == 4, f"expected boxes shape [N, 4], but got {boxes.shape}"
    if len(cls.shape) == 1:
        cls = cls[:, None]

    assert boxes.shape[0] == cls.shape[0] == scores.shape[0], f"boxes, class_id and scores shapes must be equal"

    c = cls * (0 if agnostic else max_wh)
    boxes = boxes + c
    x1 = boxes[:, 0].copy()
    y1 = boxes[:, 1].copy()
    x2 = boxes[:, 2].copy()
    y2 = boxes[:, 3].copy()

    s = scores
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    I = np.argsort(s)  # 从小到大排序索引
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size > 0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]

        xx1 = np.maximum(x1[i], x1[idx]).copy()
        yy1 = np.maximum(y1[i], y1[idx]).copy()
        xx2 = np.minimum(x2[i], x2[idx]).copy()
        yy2 = np.minimum(y2[i], y2[idx]).copy()

        w = np.maximum(0.0, xx2 - xx1 + 1).copy()
        h = np.maximum(0.0, yy2 - yy1 + 1).copy()

        inter = w * h
        if method == "Min":
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        I = I[np.where(o <= threshold)]

    pick = pick[:counter].copy()
    return pick


def bbox_iou(box1, box2, ioa=False, eps=1e-7):
    """
    Calculate the intersection over box2 area given box1 and box2. Boxes are in x1y1x2y2 format.

    Args:
        box1 (np.array): A numpy array of shape (n, 4) representing n bounding boxes.
        box2 (np.array): A numpy array of shape (m, 4) representing m bounding boxes.
        ioa (bool): Calculate inter_area/box2_area if True else return standard iou.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (np.array): A numpy array of shape (n, m) representing the intersection over box2 area.
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * (
        np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)
    ).clip(0)

    # Box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    if ioa:
        # Intersection over box2 area
        return inter_area / (box2_area + eps)
    box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area = box2_area + box1_area[:, None] - inter_area

    return inter_area / (area + eps)
