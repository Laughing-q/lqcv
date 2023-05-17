from lqcv.data import ReadVideosAndImages, ReadOneStream
import cv2

def test_img_video(source):
    reader = ReadVideosAndImages(source)
    for img, p, _ in reader:
        reader.save("sample", img)   # save to sample.jpg
        cv2.imshow('p', img)
        if cv2.waitKey(0) == ord('q'):
            break

def test_stream(source):
    reader = ReadOneStream(source)
    for img, p, _ in reader:
        reader.save("sample", img)   # save to sample.jpg
        cv2.imshow('p', img)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    # test_img_video("lqcv/assets/")
    # test_img_video("lqcv/assets/bus.jpg")
    # test_img_video("/home/laughing/Videos/test.mp4")
    test_stream("rtsp://admin:allcam123@172.16.11.133:554/Steaming/Channels/101")
