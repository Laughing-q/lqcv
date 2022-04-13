import numpy as np
import torch
import cv2
from typing import Iterable
from .bbox_utils import Boxes


class Area:
    """
    Boxes in areas or not.
    Args:
        area (List | ndarray): single area,
            if area_type is `rect`, then area is like [x1, y1, x2, y2]
            which x1,y1 is left top and x2,y2 is right bottom.
            if area_type is `polygon`, then area is like [x1, y1, x2, y2, x3, y3,...]
            which each x,y is the vertice of polygon.
        area_type (str): `rect` or `polygon`.
    """

    def __init__(self, area, area_type="rect") -> None:
        self.area = self._check(area, area_type)
        self.area_type = area_type

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

    def boxes_in_rect(self, boxes, *args, **kwargs):
        """
        Args:
            boxes (np.ndarray | torch.Tensor): xyxy format, shape: (n, 4) or (4, ).
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
            left_top = torch.as_tensor(left_top)
            right_bottom = torch.as_tensor(right_bottom)

        boxes = Boxes(boxes)
        # (m, n, 2)
        vertices = boxes.get_vertice(*args, **kwargs)
        # (m, n)
        count = (vertices > left_top).all(2) * (vertices < right_bottom).all(2)
        # (n, )
        index = count.any(0)
        return index

    def boxes_in_polygon(self, boxes, *args, **kwargs):
        """
        Args:
            boxes (np.ndarray | torch.Tensor): xyxy format, shape: (n, 4) or (4, ).
        Return:
            index (List): index of boxes which are in areas.
        """
        if len(boxes) == 0:
            return False
        boxes = Boxes(boxes)
        # (m, n, 2)
        vertices = boxes.get_vertice(*args, **kwargs)
        # torch.Tensor -> ndarray, cause `_isPointinPolygon` support numpy for now.
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.numpy()
        # (m, n)
        count = [self._isPointinPolygon(vertice, self.area) for vertice in vertices]
        index = np.asarray(count).any(0)
        return index

    def boxes_in_area(self, boxes, *args, **kwargs):
        """
        Args:
            boxes (np.ndarray | torch.Tensor): xyxy format, shape: (n, 4) or (4, ).
        Return:
            index (List): index of boxes which are in areas.
        """
        if self.area_type == "rect":
            return self.boxes_in_rect(boxes, *args, **kwargs)
        else:
            return self.boxes_in_polygon(boxes, *args, **kwargs)

    def plot(self, frame):
        if self.area_type == "rect":
            cv2.rectangle(
                frame, self.area[:2], self.area[2:], (0, 0, 255), 2, cv2.LINE_AA
            )
        else:
            cv2.polylines(
                img=frame,
                pts=[self.area],
                isClosed=True,
                color=(0, 0, 255),
                thickness=3,
            )

    def _isPointinPolygon(self, points, rangelist):  # [[0,0],[1,1],[0,1],[0,0]] [1,0.8]
        """
        :param points: numpy, 坐标xy,shape(num_points, 2)
        :param rangelist: rangelist.shape=(N+1, 2)
        :return: points in rangelist or not
        """
        if len(points) == 0:
            return []

        count = np.zeros_like(points)[:, 0]
        point1 = rangelist[0]
        for i in range(1, len(rangelist)):
            point2 = rangelist[i]
            index = ((point1[1] < points[:, 1]) * (point2[1] >= points[:, 1])) + (
                (point1[1] >= points[:, 1]) * (point2[1] < points[:, 1])
            )

            """
            求线段与射线交点 再和point的lat比较
            先取满足index条件的跑points计算,防止(point2[1] - point1[1])==0，出现nan的情况
            此方法会稍稍慢一些(大约0.0几毫秒)，但比for原本的for循环快多了
            """
            # ind记录索引
            ind = np.nonzero(index)
            points_ = points[index]
            point12lng_part = point2[0] - (point2[1] - points_[:, 1]) * (
                point2[0] - point1[0]
            ) / (point2[1] - point1[1])
            point12lng = np.zeros_like(points)[:, 0]
            point12lng[ind] = point12lng_part
            # print(ind, point12lng)

            # 判断小于<, 表示射线向x轴左边射
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

    def __init__(self, areas, area_type="rect") -> None:
        # List[area]
        self.areas = [self._check(area, area_type) for area in areas]
        assert (
            len(self.areas) > 1
        ), "you got only one area, please use single versoin `Area`."
        if area_type == "rect":
            self.areas = np.asarray(self.areas)
        self.area_type = area_type

    def boxes_in_rects(self, boxes, *args, **kwargs):
        """
        Args:
            boxes (np.ndarray | torch.Tensor): xyxy format, shape: (n, 4) or (4, ).
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
            left_top = torch.as_tensor(left_top)
            right_bottom = torch.as_tensor(right_bottom)

        boxes = Boxes(boxes)
        # (m, nb, 2)
        vertices = boxes.get_vertice(*args, **kwargs)
        # (na, m, nb)
        count = (vertices[None, :, :, :] > left_top[:, None, None, :]).all(3) * (
            vertices[None, :, :, :] < right_bottom[:, None, None, :]
        ).all(3)
        # (na, nb)
        index = count.any(1)
        return index

    def boxes_in_polygons(self, boxes, *args, **kwargs):
        """
        Args:
            boxes (np.ndarray | torch.Tensor): xyxy format, shape: (n, 4) or (4, ).
        Return:
            index (List): index of boxes which are in areas.
        """
        if len(boxes) == 0:
            return [False]
        boxes = Boxes(boxes)
        # (m, nb, 2)
        vertices = boxes.get_vertice(*args, **kwargs)
        # torch.Tensor -> ndarray, cause `_isPointinPolygon` support numpy for now.
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.numpy()
        # (na, m, nb)
        count = [
            [self._isPointinPolygon(vertice, area) for vertice in vertices]
            for area in self.areas
        ]
        # (na, nb)
        index = np.asarray(count).any(1)
        return index

    def boxes_in_areas(self, boxes, *args, **kwargs):
        """
        Args:
            boxes (np.ndarray | torch.Tensor): xyxy format, shape: (n, 4) or (4, ).
        Return:
            index (List): index of boxes which are in areas.
        """
        if self.area_type == "rect":
            return self.boxes_in_rects(boxes, *args, **kwargs)
        else:
            return self.boxes_in_polygons(boxes, *args, **kwargs)

    def plot(self, frame):
        for area in self.areas:
            if self.area_type == "rect":
                cv2.rectangle(frame, area[:2], area[2:], (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.polylines(
                    img=frame,
                    pts=[area],
                    isClosed=True,
                    color=(0, 0, 255),
                    thickness=3,
                )
