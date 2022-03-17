import glob
import os.path as osp
import cv2
import time
from pathlib import Path
from threading import Thread
from ..image.utils import IMG_FORMATS, VID_FORMATS
from ..utils.checker import check_requirements
from ..utils.general import clean_str

"""
NOTE:
    `ReadVideosAndImages` and `ReadStreams` are pure data readers without preprocessing.
"""


class ReadStreams:
    """Read Streams, modified from yolov5, support multi streams reading, but support one streams saving for now."""

    def __init__(self, sources="streams.txt", img_size=640, stride=32, auto=True):
        self.mode = "stream"
        self.img_size = img_size
        self.stride = stride

        if osp.isfile(sources):
            with open(sources, "r") as f:
                sources = [
                    x.strip() for x in f.read().strip().splitlines() if len(x.strip())
                ]
        else:
            sources = [sources]

        n = len(sources)
        self.vid_path, self.vid_writer = [None] * n, [None] * n
        self.imgs, self.fps, self.frames, self.threads = (
            [None] * n,
            [0] * n,
            [0] * n,
            [None] * n,
        )
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.auto = auto
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            print(f"{i + 1}/{n}: {s}... ", end="")
            if "youtube.com/" in s or "youtu.be/" in s:  # if source is YouTube video
                check_requirements(("pafy", "youtube_dl"))
                import pafy

                s = pafy.new(s).getbest(preftype="mp4").url  # YouTube URL
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f"Failed to open {s}"
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps[i] = (
                max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0
            )  # 30 FPS fallback
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float(
                "inf"
            )  # infinite stream fallback

            _, self.imgs[i] = cap.read()  # guarantee first frame
            self.threads[i] = Thread(
                target=self.update, args=([i, cap, s]), daemon=True
            )
            print(
                f" success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)"
            )
            self.threads[i].start()
        print("")  # newline

    def update(self, i, cap, stream):
        # Read stream `i` frames in daemon thread
        n, f, read = (
            0,
            self.frames[i],
            1,
        )  # frame number, frame array, inference every 'read' frame
        while cap.isOpened() and n < f:
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n % read == 0:
                success, im = cap.retrieve()
                if success:
                    self.imgs[i] = im
                else:
                    print(
                        "WARNING: Video stream unresponsive, please check your IP camera connection."
                    )
                    self.imgs[i] *= 0
                    cap.open(stream)  # re-open stream if signal was lost
            time.sleep(1 / self.fps[i])  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord(
            "q"
        ):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        img0 = self.imgs.copy()

        # list
        return img0, self.sources, " "

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years

    def save(self, save_path, image, i=0):
        # TODO: this won't work, cause the multi thread stuff.
        if self.vid_path[i] != save_path:  # new video
            self.vid_path[i] = save_path
            fps, w, h = 30, image.shape[1], image.shape[0]
            save_path += ".mp4"
            self.vid_writer[i] = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc(*"mp4v"), int(self.fps[i]), (w, h)
            )
        self.vid_writer[i].write(image)


class ReadVideosAndImages:
    """Read Videos and Images, modified from yolov5"""

    def __init__(self, source: str):
        p = str(Path(source).resolve())  # os-agnostic absolute path
        if "*" in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif osp.isdir(p):
            files = sorted(glob.glob(osp.join(p, "*.*")))  # dir
        elif osp.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f"ERROR: {p} does not exist")

        # random.shuffle(files)
        images = [x for x in files if x.split(".")[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split(".")[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.vid_path, self.vid_writer = None, None
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = "image"
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, (
            f"No images or videos found in {p}. "
            f"Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}"
        )

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = "video"
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            s = f"video {self.count + 1}/{self.nf} "
        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, "Image Not Found " + path
            s = f"image {self.count}/{self.nf} "

        return img0, path, s

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def save(self, save_path, image):
        if self.mode == "image":
            cv2.imwrite(save_path, image)
        else:  # 'video' or 'stream'
            if self.vid_path != save_path:  # new video
                self.vid_path = save_path
                if isinstance(self.vid_writer, cv2.VideoWriter):
                    self.vid_writer.release()  # release previous video writer
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.vid_writer = cv2.VideoWriter(
                    save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                )
            self.vid_writer.write(image)

    def __len__(self):
        return self.nf  # number of files


def create_reader(source: str):
    """This is for data(video, webcam, image, image_path) reading in inference.
    Args:
        source(str): data source, could be a video,image,dir or webcam.
    Return:
        reader(ReadVideosAndImages | ReadStreams): data reader.
        webcam(bool): the source is webcam or not.
    """
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".txt") or (is_url and not is_file)
    return ReadStreams(source) if webcam else ReadVideosAndImages(source), webcam


if __name__ == "__main__":
    test = create_reader(source="/d/九江/playphone/20211223/imgs")
    for img, p, s in test:
        print(s, p)
        cv2.imshow("p", img)
        if cv2.waitKey(1) == ord("q"):
            break
