from typing import List
from lqcv.utils import ops


__all__ = ["Boxes"]
_formats = ["xyxy", "xywh", "ltwh"]


class Boxes:
    """
    Args:
        boxes (np.ndarray | torch.Tensor): boxes.
        format (str): box format, should be xyxy or xywh or ltwh.
    """

    def __init__(self, boxes, format="xyxy"):
        assert format in _formats, f"Invalid bounding box format: {format}, format must be one of {_formats}"
        boxes = boxes[None, :] if boxes.ndim == 1 else boxes
        assert boxes.ndim == 2
        assert boxes.shape[1] == 4
        # (n, 4)
        self._boxes = boxes
        self.format = format

    def convert(self, format):
        """Converts bounding box format from one type to another."""
        assert format in _formats, f"Invalid bounding box format: {format}, format must be one of {_formats}"
        if self.format == format:
            return
        elif self.format == "xyxy":
            boxes = ops.xyxy2xywh(self._boxes) if format == "xywh" else ops.xyxy2ltwh(self._boxes)
        elif self.format == "xywh":
            boxes = ops.xywh2xyxy(self._boxes) if format == "xyxy" else ops.xywh2ltwh(self._boxes)
        else:
            boxes = ops.ltwh2xyxy(self._boxes) if format == "xyxy" else ops.ltwh2xywh(self._boxes)
        self._boxes = boxes
        self.format = format

    @classmethod
    def stack(cls, boxes_list: List["Boxes"], dim=0):
        assert isinstance(boxes_list, (list, tuple))

        if len(boxes_list) == 1:
            return boxes_list[0]
        return cls(ops.stack([b.bboxes for b in boxes_list], dim=dim))

    @classmethod
    def cat(cls, boxes_list: List["Boxes"], dim=0):
        assert isinstance(boxes_list, (list, tuple))

        if len(boxes_list) == 1:
            return boxes_list[0]
        return cls(ops.cat([b.data for b in boxes_list], dim=dim))

    @property
    def lt(self):
        """left top, (n, 2)"""
        self.convert("xyxy")
        return self._boxes[:, :2]

    @property
    def rt(self):
        """right top, (n, 2)"""
        self.convert("xyxy")
        return ops.stack([self._boxes[:, 2], self._boxes[:, 1]], 1)

    @property
    def lb(self):
        """left bottom, (n, 2)"""
        self.convert("xyxy")
        return ops.stack([self._boxes[:, 0], self._boxes[:, 3]], 1)

    @property
    def rb(self):
        self.convert("xyxy")
        """right bottom, (n, 2)"""
        return self._boxes[:, 2:]

    @property
    def center(self):
        """center, (n, 2)"""
        self.convert("xywh")
        return self._boxes[:, :2]

    @property
    def areas(self):
        format = self.format
        if format != "xyxy":
            self.convert("xyxy")
        areas = (self._boxes[:, 2] - self._boxes[:, 0]) * (self._boxes[:, 3] - self._boxes[:, 1])
        # convert back to the original format
        if format != "xyxy":
            self.convert(format)
        return areas

    @property
    def data(self):
        return self._boxes

    @property
    def shape(self):
        return self._boxes.shape

    def get_coords(self, filter=None, frame=None):
        """Get the coordinates of self.boxes.
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
            coordinates = ops.stack([self.lt, self.lb, self.rt, self.rb], 0)
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
            coordinates = coords[0][None, :, :] if len(coords) == 1 else ops.stack(coords, 0)
        if frame is not None:
            import cv2

            for coords in coordinates:
                for coord in coords:
                    cv2.circle(frame, (int(coord[0]), int(coord[1])), 1, (0, 0, 255), 5)
        return coordinates

    def __len__(self):
        return len(self._boxes)

    def __getitem__(self, index) -> "Boxes":
        """
        Retrieve a specific bounding box or a set of bounding boxes using indexing.

        Args:
            index (int, slice, or np.ndarray): The index, slice, or boolean array to select
                                               the desired bounding boxes.

        Returns:
            Bboxes: A new Bboxes object containing the selected bounding boxes.

        Raises:
            AssertionError: If the indexed bounding boxes do not form a 2-dimensional matrix.

        Note:
            When using boolean indexing, make sure to provide a boolean array with the same
            length as the number of bounding boxes.
        """
        b = self._boxes[index]
        return Boxes(b, self.format)

    @property
    def data(self):
        """The original data."""
        return self._boxes

    @classmethod
    def iou(cls, box1, box2, ioa=False):
        """Calculate iou.

        Args:
            box1 (np.ndarray | Boxes): The box with shape (n, 4).
            box2 (np.ndarray | Boxes): The box with shape (m, 4).
            ioa (bool): Calculate inter_area/box2_area if True else return standard iou.
        Returns:
            np.ndarray, with shape (n, m).
        """
        if isinstance(box1, Boxes) and isinstance(box2, Boxes):
            box1.convert("xyxy")
            box2.convert("xyxy")
            return ops.bbox_iou(box1.data, box2.data, ioa=ioa)
        return ops.bbox_iou(box1, box2, ioa=ioa)
