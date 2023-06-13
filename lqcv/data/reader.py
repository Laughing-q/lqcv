import glob
import os.path as osp
import cv2
import time
from pathlib import Path
from threading import Thread
from .utils import IMG_FORMATS, VID_FORMATS


class ReadStreams:
    """Read Streams, modified from yolov5, support multi streams reading, but support one streams saving for now."""

    def __init__(self, sources="streams.txt"):
        self.mode = "stream"

        if osp.isfile(sources):
            with open(sources, "r") as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
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
        self.sources = sources  # clean source names for later
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            print(f"{i + 1}/{n}: {s}... ", end="")
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f"Failed to open {s}"
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps[i] = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0  # 30 FPS fallback
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float(
                "inf"
            )  # infinite stream fallback

            _, self.imgs[i] = cap.read()  # guarantee first frame
            self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
            print(f" success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
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
                    print("WARNING: Video stream unresponsive, please check your IP camera connection.")
                    self.imgs[i] *= 0
                    cap.open(stream)  # re-open stream if signal was lost
            time.sleep(1 / self.fps[i])  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord("q"):  # q to quit
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


class ReadOneStream:
    """Read one Stream, modified from yolov5, support one streams reading and saving."""

    def __init__(self, source, vid_stride=1, imgsz=None, im_only=False):
        """ReadOneStream

        Args:
            source (str): Source, could be a folder or a direct file.
            vid_stride (int | optional): Video stride.
            imgsz (tuple | optional): Image size, (height, width)
            im_only (bool | optional): Whether to return image only, or it'll return
                image, path and description.
        """
        self.mode = "stream"

        self.vid_path, self.vid_writer = None, None
        self.fps, self.frames = 0, 0
        self.source = source  # clean source names for later
        self.vid_stride = vid_stride
        self.imgsz = imgsz
        self.im_only = im_only

        # Start thread to read frames from video stream
        print(f"{1}/{1}: {source}... ", end="")
        source = eval(source) if source.isnumeric() else source  # i.e. s = '0' local webcam
        self.cap = cv2.VideoCapture(source)
        assert self.cap.isOpened(), f"Failed to open {source}"
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = max(self.cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0  # 30 FPS fallback
        self.frames = max(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float(
            "inf"
        )  # infinite stream fallback

        print(f" success ({self.frames} frames {w}x{h} at {self.fps:.2f} FPS)")
        print("")  # newline

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        for _ in range(self.vid_stride):
            self.cap.grab()
        success, img0 = self.cap.retrieve()
        if not success or cv2.waitKey(1) == ord("q"):  # q to quit
            self.count += 1
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        if self.imgsz:
            h, w = self.imgsz
            img0 = cv2.resize(img0, (w, h))
        return img0 if self.im_only else (img0, self.source, " ")

    def __len__(self):
        return 1

    def save(self, save_path, image):
        """save video.

        Args:
            save_path (str): Save path, with suffix(`.avi`, `.mp4`) or not.
            image (nd.ndarray): The image/frame.
        """
        save_path = f"{save_path}.mp4" if Path(save_path).suffix[1:] not in VID_FORMATS else save_path
        if self.vid_path != save_path:  # new video
            self.vid_path = save_path
            w, h = image.shape[1], image.shape[0]
            self.vid_writer = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc(*"mp4v"), int(self.fps), (w, h)
            )
        self.vid_writer.write(image)


class ReadVideosAndImages:
    """Read Videos and Images, modified from yolov5"""

    def __init__(self, source: str, vid_stride=1, imgsz=None, im_only=False):
        """ReadVideosAndImages

        Args:
            source (str): Source, could be a folder or a direct file.
            vid_stride (int | optional): Video stride.
            imgsz (tuple | optional): Image size, (height, width)
            im_only (bool | optional): Whether to return image only, or it'll return
                image, path and description.
        """
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
        self.vid_stride = vid_stride
        self.mode = "image"
        self.imgsz = imgsz
        self.im_only = im_only
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
            for _ in range(self.vid_stride):
                self.cap.grab()
            success, img0 = self.cap.retrieve()
            if self.frame >= self.frames:
            # if not success:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    success, img0 = self.cap.read()

            self.frame += self.vid_stride
            s = f"video {self.count + 1}/{self.nf} {self.frame}/{self.frames}"
        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, "Image Not Found " + path
            s = f"image {self.count}/{self.nf} "

        if self.imgsz:
            h, w = self.imgsz
            img0 = cv2.resize(img0, (w, h))
        return img0 if self.im_only else (img0, path, s)

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        try:
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        except:
            print("Warning: get inf fps, hand-coded it to 25.")
            self.fps = 25

    def save(self, save_path, image):
        """save image or video.

        Args:
            save_path (str): Save path, with suffix(`.jpg`, `.mp4`) or not.
            image (nd.ndarray): The image/frame.
        """
        if self.mode == "image":
            save_path = f"{save_path}.jpg" if Path(save_path).suffix[1:] not in IMG_FORMATS else save_path
            cv2.imwrite(save_path, image)
        else:  # 'video' or 'stream'
            save_path = f"{save_path}.mp4" if Path(save_path).suffix[1:] not in VID_FORMATS else save_path
            if self.vid_path != save_path:  # new video
                self.vid_path = save_path
                if isinstance(self.vid_writer, cv2.VideoWriter):
                    self.vid_writer.release()  # release previous video writer
                w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
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
    return ReadOneStream(source) if webcam else ReadVideosAndImages(source), webcam
