import numpy as np
import torch
import cv2
from typing import Iterable
from .bbox import Boxes

__all__ = ["Area", "Areas"]


class Area:
    """
    Boxes in areas or not.
    Args:
        area (List | ndarray): single area,
            if area_type is `rect`, then area is like [x1, y1, x2, y2]
            which x1,y1 is left top and x2,y2 is right bottom.
            if area_type is `polygon`, then area is like [x1, y1, x2, y2, x3, y3,...]
            which each x,y is the vertice of polygon.
        atype (str): area type, `rect` or `polygon`.
    """

    def __init__(self, area, atype="rect") -> None:
        self.area = self._check(area, atype)
        self.atype = atype

    def _check(self, area, area_type):
        assert area_type in ["rect", "polygon"]
        assert isinstance(area, Iterable)
        if area_type == "rect":
            assert len(area) == 4
            x1, y1, x2, y2 = area
            assert x2 >= x1 and y2 >= y1
            return area
        else:
            assert len(area) % 2 == 0
            # Polygon vertices need to be connected end to end
            area = area + area[:2]
            area = np.asarray(area)
            area = area.reshape(len(area) // 2, 2)
            return area

    def boxes_in_rect(self, boxes, filter=None, frame=None):
        """
        Args:
            boxes (np.ndarray | torch.Tensor): xyxy format, shape: (n, 4) or (4, ).
            filter (str | List[str] | [optional]): include `lt`,
                `lb`, `rt`, `rb` or `center`. If None(default), then it's
                ["lt", "lb", "rt", "rb"].
            frame (ndarray): image for visualization.
        Return:
            index (np.ndarray | torch.Tensor): same type and length with boxes, index
                of boxes which are in areas.
        """
        if len(boxes) == 0:
            return False

        left_top = self.area[:2]
        right_bottom = self.area[2:]
        # deal with torch.Tensor
        if isinstance(boxes, torch.Tensor):
            left_top = torch.as_tensor(left_top, device=boxes.device)
            right_bottom = torch.as_tensor(right_bottom, device=boxes.device)

        boxes = Boxes(boxes)
        # (m, n, 2)
        coordinates = boxes.get_coords(filter=filter, frame=frame)
        # (m, n)
        count = (coordinates > left_top).all(2) * (coordinates < right_bottom).all(2)
        # (n, )
        index = count.any(0)
        return index

    def boxes_in_polygon(self, boxes, filter=None, frame=None):
        """
        Args:
            boxes (np.ndarray | torch.Tensor): xyxy format, shape: (n, 4) or (4, ).
            filter (str | List[str] | [optional]): include `lt`,
                `lb`, `rt`, `rb` or `center`. If None(default), then it's
                ["lt", "lb", "rt", "rb"].
            frame (ndarray): image for visualization.
        Return:
            index (List): index of boxes which are in areas.
        """
        if len(boxes) == 0:
            return False
        boxes = Boxes(boxes)
        # (m, n, 2)
        coordinates = boxes.get_coords(filter=filter, frame=frame)
        # torch.Tensor -> ndarray, cause `_isPointinPolygon` support numpy for now.
        if isinstance(coordinates, torch.Tensor):
            coordinates = coordinates.cpu().numpy()
        # (m, n)
        count = [Area.isPointinPolygon(vertice, self.area) for vertice in coordinates]
        index = np.asarray(count).any(0)
        return index

    def filter_boxes(self, boxes, filter=None, frame=None):
        """
        Args:
            boxes (np.ndarray | torch.Tensor): xyxy format, shape: (n, 4) or (4, ).
            filter (str | List[str] | [optional]): include `lt`,
                `lb`, `rt`, `rb` or `center`. If None(default), then it's
                ["lt", "lb", "rt", "rb"].
            frame (ndarray): image for visualization.
        Return:
            index (List): index of boxes which are in areas.
        """
        if self.atype == "rect":
            return self.boxes_in_rect(boxes, filter=filter, frame=frame)
        else:
            return self.boxes_in_polygon(boxes, filter=filter, frame=frame)

    def plot(self, frame):
        if self.atype == "rect":
            cv2.rectangle(frame, self.area[:2], self.area[2:], (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.polylines(
                img=frame,
                pts=[self.area],
                isClosed=True,
                color=(0, 0, 255),
                thickness=3,
            )

    @staticmethod
    def isPointinPolygon(points, rangelist):  # [[0,0],[1,1],[0,1],[0,0]] [1,0.8]
        """

        Args:
            points (np.ndarry): (n, 2), xy format.
            rangelist (list): polygon coordinates.

        Returns:
            points in rangelist or not
            
        """
        if len(points) == 0:
            return []

        count = np.zeros_like(points[:, 0])
        point1 = rangelist[0]
        for i in range(1, len(rangelist)):
            point2 = rangelist[i]
            index = ((point1[1] < points[:, 1]) * (point2[1] >= points[:, 1])) + (
                (point1[1] >= points[:, 1]) * (point2[1] < points[:, 1])
            )
            ind = np.nonzero(index)
            points_ = points[index]
            point12lng_part = point2[0] - (point2[1] - points_[:, 1]) * (point2[0] - point1[0]) / (
                point2[1] - point1[1]
            )
            point12lng = np.zeros_like(points[:, 0])
            point12lng[ind] = point12lng_part
            # print(ind, point12lng)

            count[(point12lng < points[:, 0]) * point12lng != 0] += 1
            point1 = point2

        return np.asarray(count % 2, dtype=np.bool)


class Areas(Area):
    """
    Args:
        areas (List | ndarray): multi areas,
            if area_type is `rect`, then area is like List[[x1, y1, x2, y2], [...]]
            which x1,y1 is left top and x2,y2 is right bottom.
            if area_type is `polygon`, then area is like List[[x1, y1, x2, y2, x3, y3,...], [...]]
            which each x,y is the vertice of polygon.
        area_type (str): `rect` or `polygon`.
    """

    def __init__(self, areas, atype="rect") -> None:
        # List[area]
        self.areas = [self._check(area, atype) for area in areas]
        assert len(self.areas) > 1, "you got only one area, please use single versoin `Area`."
        if atype == "rect":
            self.areas = np.asarray(self.areas)
        self.atype = atype

    def boxes_in_rects(self, boxes, filter=None, frame=None):
        """
        Args:
            boxes (np.ndarray | torch.Tensor): xyxy format, shape: (n, 4) or (4, ).
            filter (str | List[str] | [optional]): include `lt`,
                `lb`, `rt`, `rb` or `center`. If None(default), then it's
                ["lt", "lb", "rt", "rb"].
            frame (ndarray): image for visualization.
        Return:
            index (np.ndarray | torch.Tensor): same type and length with boxes, index
                of boxes which are in areas.
        """
        if len(boxes) == 0:
            return [False]

        # (na, 2)
        left_top = self.areas[:, :2]
        right_bottom = self.areas[:, 2:]
        # deal with torch.Tensor
        if isinstance(boxes, torch.Tensor):
            left_top = torch.as_tensor(left_top, device=boxes.device)
            right_bottom = torch.as_tensor(right_bottom, device=boxes.device)

        boxes = Boxes(boxes)
        # (m, nb, 2)
        vertices = boxes.get_coords(filter=filter, frame=frame)
        # (na, m, nb)
        count = (vertices[None, :, :, :] > left_top[:, None, None, :]).all(3) * (
            vertices[None, :, :, :] < right_bottom[:, None, None, :]
        ).all(3)
        # (na, nb)
        index = count.any(1)
        return index

    def boxes_in_polygons(self, boxes, filter=None, frame=None):
        """
        Args:
            boxes (np.ndarray | torch.Tensor): xyxy format, shape: (n, 4) or (4, ).
            filter (str | List[str] | [optional]): include `lt`,
                `lb`, `rt`, `rb` or `center`. If None(default), then it's
                ["lt", "lb", "rt", "rb"].
            frame (ndarray): image for visualization.
        Return:
            index (np.ndarray): index of boxes which are in areas.
        """
        if len(boxes) == 0:
            return [False]
        boxes = Boxes(boxes)
        # (m, nb, 2)
        vertices = boxes.get_coords(filter=filter, frame=frame)
        # NOTE: torch.Tensor -> ndarray, cause `_isPointinPolygon` support numpy for now.
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.cpu().numpy()
        # (na, m, nb)
        count = [
            [Area.isPointinPolygon(vertice, area) for vertice in vertices] for area in self.areas
        ]
        # (na, nb)
        index = np.asarray(count).any(1)
        return index

    def filter_boxes(self, boxes, filter=None, frame=None):
        """
        Args:
            boxes (np.ndarray | torch.Tensor): xyxy format, shape: (n, 4) or (4, ).
            filter (str | List[str] | [optional]): include `lt`,
                `lb`, `rt`, `rb` or `center`. If None(default), then it's
                ["lt", "lb", "rt", "rb"].
            frame (ndarray): image for visualization.
        Return:
            index (np.ndarray | torch.Tensor): index of boxes which are in areas.
        """
        if self.atype == "rect":
            return self.boxes_in_rects(boxes, filter=filter, frame=frame)
        else:
            return self.boxes_in_polygons(boxes, filter=filter, frame=frame)

    def plot(self, frame):
        for area in self.areas:
            if self.atype == "rect":
                cv2.rectangle(frame, area[:2], area[2:], (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.polylines(
                    img=frame,
                    pts=[area],
                    isClosed=True,
                    color=(0, 0, 255),
                    thickness=3,
                )
