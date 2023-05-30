from lqcv.data import ReadVideosAndImages, ReadOneStream
from lqcv.utils.plot import waitKey
import cv2

def test_img_video(source):
    reader = ReadVideosAndImages(source)
    # reader = ReadVideosAndImages(source, vid_stride=3)
    for img, p, s in reader:
        if img is None:
            continue
        print(s)
        reader.save("sample", img)   # save to sample.jpg
        cv2.imshow('p', img)
        if cv2.waitKey(1) == ord('q'):
            break

def test_stream(source):
    reader = ReadOneStream(source, vid_stride=3)
    for img, p, _ in reader:
        reader.save("sample", img)   # save to sample.jpg
        cv2.imshow('p', img)
        if cv2.waitKey(1) == ord('q'):
            break

def test_img_video2(source):
    reader = ReadVideosAndImages(source, imgsz=(640, 640), im_only=True)
    for img in reader:
        if img is None:
            continue
        cv2.imshow('p', img)
        if waitKey(1) == ord('q'):
            break

def test_stream2(source):
    reader = ReadOneStream(source, imgsz=(640, 640), im_only=True)
    for img in reader:
        if img is None:
            continue
        cv2.imshow('p', img)
        if waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    # test_img_video("lqcv/assets/")
    # test_img_video("lqcv/assets/bus.jpg")
    # test_img_video("/home/laughing/Videos/test.mp4")
    # test_stream("rtsp://admin:allcam123@172.16.11.133:554/Steaming/Channels/101")
    # test_img_video("/d/dataset/长沙/videos/advertise_0104/佳欣小学保家村委会门口（四期）_南_20230104141959_001.mp4")

    test_img_video2("/home/laughing/Videos/test.mp4")
    # test_stream2("rtsp://admin:allcam123@172.16.11.133:554/Steaming/Channels/101")
