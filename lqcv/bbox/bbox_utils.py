from typing import List
from lqcv.utils import ops
import numpy as np
import torch


class Boxes:
    """
    Args:
        boxes (np.ndarray | torch.Tensor): boxes, xyxy format.
    """

    def __init__(self, boxes) -> None:
        boxes = boxes[None, :] if boxes.ndim == 1 else boxes
        assert boxes.ndim == 2
        assert boxes.shape[1] == 4
        # n, 4
        self.boxes = boxes

    @classmethod
    def stack(cls, boxes_list, dim=0):
        assert isinstance(boxes_list, (list, tuple))

        if len(boxes_list) == 1:
            return boxes_list[0]
        return cls(ops.stack([b.bboxes for b in boxes_list], dim=dim))

    @classmethod
    def cat(cls, boxes_list, dim=0):
        assert isinstance(boxes_list, (list, tuple))

        if len(boxes_list) == 1:
            return boxes_list[0]
        return cls(ops.cat([b.bboxes for b in boxes_list], dim=dim))

    @property
    def lt(self):
        """left top, (n, 2)"""
        return self.boxes[:, :2]

    @property
    def rt(self):
        """right top, (n, 2)"""
        return ops.stack([self.boxes[:, 2], self.boxes[:, 1]], 1)

    @property
    def lb(self):
        """left bottom, (n, 2)"""
        return ops.stack([self.boxes[:, 0], self.boxes[:, 3]], 1)

    @property
    def rb(self):
        """right bottom, (n, 2)"""
        return self.boxes[:, 2:]

    @property
    def center(self):
        """center, (n, 2)"""
        x_center = (self.boxes[:, 0] + self.boxes[:, 2]) / 2
        y_center = (self.boxes[:, 1] + self.boxes[:, 3]) / 2
        return ops.stack([x_center, y_center], 1)

    @property
    def areas(self):
        areas = (self.boxes[:, 2] - self.boxes[:, 0]) * (
            self.boxes[:, 3] - self.boxes[:, 1]
        )
        return areas

    def get_vertice(self, filter=None, frame=None):
        """Get the vertice of self.boxes.
        Args:
            filter (str | List[str] | [optional]): include `lt`,
                `lb`, `rt`, `rb` or `center`, if None, return
                `lt` + `lb` + `rt` + `rb`.
            frame (ndarray): image for visualization.
        Return:
            same type with self.boxes, shape: (m, n, 2),
                n is number of boxes, m if number of vertices.
        """
        if filter is None:
            # NOTE: exclude center
            vertices = ops.stack([self.lt, self.lb, self.rt, self.rb], 0)
        else:
            if isinstance(filter, str):
                filter = [filter]
            coords = []
            for f in filter:
                f in ["lt", "lb", "rt", "rb", "center"]
                if f == "lt":
                    coords.append(self.lt)
                elif f == "lb":
                    coords.append(self.lb)
                elif f == "rt":
                    coords.append(self.rt)
                elif f == "rb":
                    coords.append(self.rb)
                else:
                    coords.append(self.center)
            vertices = (
                coords[0][None, :, :] if len(coords) == 1 else ops.stack(coords, 0)
            )
        if frame is not None:
            import cv2

            for coords in vertices.tolist():
                for coord in coords:
                    cv2.circle(frame, (int(coord[0]), int(coord[1])), 1, (0, 0, 255), 5)
        return vertices
