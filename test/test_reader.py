from lqcv.data import create_reader
import cv2

dataset, _ = create_reader('rtsp://admin:shtf123456@192.168.1.231:554/h264/ch1/main/av_stream')

base = None
for img, p, _ in dataset:
    image = img[0]
    dataset.save(save_path='./output', image=image)
    # save_path = "output.mp4"
    # if save_path != base:
    #     fps, w, h = 30, image.shape[1], image.shape[0]
    #     vid_writer = cv2.VideoWriter(
    #         save_path, cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (w, h)
    #     )
    # vid_writer.write(image)
    cv2.imshow('p', image)
    if cv2.waitKey(1) == ord('q'):
        break
