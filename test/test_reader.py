from lqcv.data import create_reader
import cv2

dataset, _ = create_reader('0')

base = None
for img, p, _ in dataset:
    image = img
    dataset.save(save_path='./output.mp4', image=image)
    # save_path = "output.mp4"
    # if save_path != base:
    #     base = save_path
    #     fps, w, h = 30, image.shape[1], image.shape[0]
    #     vid_writer = cv2.VideoWriter(
    #         save_path, cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (w, h)
    #     )
    # vid_writer.write(image)
    cv2.imshow('p', image)
    if cv2.waitKey(1) == ord('q'):
        break
