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

        # (n, )
        self.x1 = boxes[:, 0]
        self.y1 = boxes[:, 1]
        self.x2 = boxes[:, 2]
        self.y2 = boxes[:, 3]
        # n, 4
        self.boxes = boxes

    def stack(self, *args, **kwargs):
        return (
            torch.stack(*args, **kwargs)
            if isinstance(self.boxes, torch.Tensor)
            else np.stack(*args, **kwargs)
        )

    def cat(self):
        return torch.cat if isinstance(self.boxes, torch.Tensor) else np.concatenate

    @property
    def lt(self):
        """left top, (n, 2)"""
        return self.stack([self.x1, self.y1], 1)

    @property
    def rt(self):
        """right top, (n, 2)"""
        return self.stack([self.x2, self.y1], 1)

    @property
    def lb(self):
        """left bottom, (n, 2)"""
        return self.stack([self.x1, self.y2], 1)

    @property
    def rb(self):
        """right bottom, (n, 2)"""
        return self.stack([self.x2, self.y2], 1)

    @property
    def center(self):
        """center, (n, 2)"""
        x_center = (self.x1 + self.x2) / 2
        y_center = (self.y1 + self.y2) / 2
        return self.stack([x_center, y_center], 1)

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
            vertices = self.stack([self.lt, self.lb, self.rt, self.rb], 0)
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
                coords[0][None, :, :] if len(coords) == 1 else self.stack(coords, 0)
            )
        if frame is not None:
            for coords in vertices.tolist():
                for coord in coords:
                    cv2.circle(frame, (int(coord[0]), int(coord[1])), 1, (0, 0, 255), 5)
        return vertices
