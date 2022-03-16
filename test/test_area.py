from lqcv.bbox.areas import Area, Areas
from lqcv.data import create_reader
from yolov5.core import Yolov5
from tqdm import tqdm
import cv2


def test_one_rect(rect, detector, dataset):
    area = Area(rect, area_type="rect")
    for img, _, _ in tqdm(dataset, total=dataset.frames):
        outputs = detector.inference(img, classes=[0])
        boxes = outputs[0][:, :4].cpu()
        index = area.boxes_in_area(boxes, frame=img, filter="center")
        img = detector.visualize(img, outputs[0][index])
        area.plot(img)
        cv2.imshow("p", cv2.resize(img, (1280, 720)))
        if cv2.waitKey(1) == ord("q"):
            break


def test_multi_rects(rects, detector, dataset):
    areas = Areas(rects, area_type="rect")
    for img, _, _ in tqdm(dataset, total=dataset.frames):
        outputs = detector.inference(img, classes=[0])
        boxes = outputs[0][:, :4].cpu()
        index = areas.boxes_in_areas(boxes, frame=img, filter="center")
        for i in index:
            img = detector.visualize(img, outputs[0][i])
        areas.plot(img)
        cv2.imshow("p", cv2.resize(img, (1280, 720)))
        if cv2.waitKey(1) == ord("q"):
            break


def test_one_polygon(polygon, detector, dataset):
    area = Area(polygon, area_type="polygon")
    for img, _, _ in tqdm(dataset, total=dataset.frames):
        outputs = detector.inference(img, classes=[0])
        boxes = outputs[0][:, :4].cpu()
        index = area.boxes_in_area(boxes, frame=img, filter="lb")
        img = detector.visualize(img, outputs[0][index])
        area.plot(img)
        cv2.imshow("p", cv2.resize(img, (1280, 720)))
        if cv2.waitKey(1) == ord("q"):
            break


def test_multi_polygons(polygons, detector, dataset):
    areas = Areas(polygons, area_type="polygon")
    for img, _, _ in tqdm(dataset, total=dataset.frames):
        outputs = detector.inference(img, classes=[0])
        boxes = outputs[0][:, :4].cpu()
        index = areas.boxes_in_areas(boxes, frame=img, filter="center")
        for i in index:
            img = detector.visualize(img, outputs[0][i])
        areas.plot(img)
        cv2.imshow("p", cv2.resize(img, (1280, 720)))
        if cv2.waitKey(1) == ord("q"):
            break


if __name__ == "__main__":
    rect = [545, 248, 1414, 616]
    polygon = [559, 245, 1035, 242, 1136, 668, 253, 720]

    rects = [[545, 248, 1414, 616], [1325, 216, 1806, 910], [97, 747, 945, 987]]
    polygons = [
        [559, 245, 1035, 242, 1136, 668, 253, 720],
        [1399, 193, 1777, 462, 1536, 715, 1044, 695],
        [115, 714, 512, 720, 821, 883, 502, 1021, 131, 974],
    ]

    detector = Yolov5(weights="./weights/yolov5m.pt", device=0)
    dataset, _ = create_reader(source="./test_videos/test.mp4")

    # test_one_rect(rect, detector, dataset)
    # test_one_polygon(polygon, detector, dataset)

    # test_multi_rects(rects, detector, dataset)
    test_multi_polygons(polygons, detector, dataset)
