from lqcv.bbox import Area, Areas
from ultralytics import YOLO
import cv2


def test_one(results, area, atype="rect", numpy=False):
    area = Area(area, atype=atype)
    for result in results:
        boxes = result.boxes.xyxy
        boxes = boxes.cpu().numpy() if numpy else boxes
        img = result.orig_img
        index = area.filter_boxes(boxes, frame=img, filter="center")
        plot_img = result[index].plot()
        area.plot(plot_img)
        cv2.imshow("p", cv2.resize(plot_img, (1280, 720)))
        if cv2.waitKey(1) == ord("q"):
            break


def test_multi(results, areas, atype="rect", numpy=False):
    areas = Areas(areas, atype=atype)
    for result in results:
        boxes = result.boxes.xyxy
        boxes = boxes.cpu().numpy() if numpy else boxes
        img = result.orig_img
        index = areas.filter_boxes(boxes, frame=img, filter="center")
        for i in index:
            img = result[i].plot(img=img)
        areas.plot(img)
        cv2.imshow("p", cv2.resize(img, (1280, 720)))
        if cv2.waitKey(1) == ord("q"):
            break


if __name__ == "__main__":
    # NOTE: example rect and polygon for 1920x1080 video.
    rect = [545, 248, 1414, 616]
    polygon = [559, 245, 1035, 242, 1136, 668, 253, 720]

    rects = [[545, 248, 1414, 616], [1325, 216, 1806, 910], [97, 747, 945, 987]]
    polygons = [
        [559, 245, 1035, 242, 1136, 668, 253, 720],
        [1399, 193, 1777, 462, 1536, 715, 1044, 695],
        [115, 714, 512, 720, 821, 883, 502, 1021, 131, 974],
    ]

    model = YOLO("/home/laughing/codes/ultralytics/weights/yolov8n.pt")
    results = model.predict("/home/laughing/Videos/test_person.mp4", stream=True, classes=0)
    # test_one(results, rect, atype="rect")
    # test_one(results, polygon, atype="polygon")
    # test_multi(results, rects, atype="rect")
    # test_multi(results, polygons, atype="polygon")

    test_one(results, rect, atype="rect", numpy=True)
    test_one(results, polygon, atype="polygon", numpy=True)
    test_multi(results, rects, atype="rect", numpy=True)
    test_multi(results, polygons, atype="polygon", numpy=True)
