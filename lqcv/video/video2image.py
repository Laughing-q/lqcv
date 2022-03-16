import cv2
import time
from tqdm import tqdm
import os
from pathlib import Path

"""
使用opencv捕获摄像头,并保存为视频
"""
"""
mode each:每个视频的图片单独放一个文件夹
mode all：所有视频的图片都放在一个文件夹
"""
mode = "all"
interval = 6  # 多少秒保存一次
count_only = False

vid_formats = [
    ".mov",
    ".avi",
    ".mp4",
    ".mpg",
    ".mpeg",
    ".m4v",
    ".wmv",
    ".mkv",
    ".ts",
    ".hmv",
    ".vdo",
]
# video_dir可以是视频，也可以是装有多个视频的文件夹
video_dir = "/d/九江/passengers/other_suit/大堂经理_datasets1"
save_dir = "/d/九江/passengers/other_suit/images1"

rename = False

tail = ""

resize_frame = False
resize_shape = (1280, 720)

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

if os.path.isfile(video_dir):
    videos = [video_dir]
else:
    videos = sorted(
        [
            video
            for video in os.listdir(video_dir)
            if os.path.splitext(video)[-1].lower() in vid_formats
        ]
    )
print(videos)

nf = len(videos)

total_pic = 0
new_name = -1

for i, video in enumerate(videos):
    new_name += 1
    if mode == "each":
        if not os.path.exists(os.path.join(save_dir, video)):
            os.mkdir(os.path.join(save_dir, video))
    # print(video_dir, video)
    # print(os.path.join(video_dir, video))
    # exit()
    use_video = os.path.join(video_dir, video)
    cap = cv2.VideoCapture(use_video)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(nframes)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps读出来是inf，int(inf)会报错
    try:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
    except:
        fps = 25
    if count_only:
        if nframes > 0:
            total_pic += nframes / (fps * interval if fps <= 100 else 25 * interval)
        else:
            print(f"nframes should > 0, but {use_video} got {nframes} nframes.")
        continue
    # print(fps)
    # continue
    # size=(1280,720)

    # video_name = os.path.split(video)[0]
    # print(video_name)
    video_name = Path(use_video).name.split(".")[0] if not rename else new_name

    print(video_name)
    # continue

    frame_num = -1

    while cap.isOpened():
        frame_num += 1
    # pbar = tqdm(range(nframes), total=nframes)
    # for frame_num in pbar:
        # print('frame_num:', frame_num)

        ret, frame = cap.read()
        if resize_frame:
            frame = cv2.resize(frame, resize_shape)
            w, h = resize_shape

        if not ret:
            print("==============视频播放完毕=============")
            break

        # if frame_num % (fps * interval) == 0:
        if (
            frame_num % (fps * interval if fps <= 30 else 25 * interval) == 0
        ):  # 有时候会有读取帧数为1000的情况
            # if 1:  # 有时候会有读取帧数为1000的情况
            # frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            total_pic += 1
            log = "video:%g/%g frame:(%g/%g) W×H:(%g/%g) fps:%g %s (total:%g)" % (
                i + 1,
                nf,
                frame_num,
                nframes,
                w,
                h,
                fps,
                use_video,
                total_pic,
            )
            print(log)
            # pbar.desc = log
            if mode == "each":
                cv2.imwrite(
                    os.path.join(
                        os.path.join(save_dir, video),
                        f"{video_name}_{frame_num}_{tail}.jpg",
                    ),
                    frame,
                )
            else:
                # print(os.path.join(save_dir, f'{video_name}_{frame_num}_{tail}.jpg'))
                # print(save_dir)
                # print(video_name)
                cv2.imwrite(
                    os.path.join(save_dir, f"{video_name}_{frame_num}_{tail}.jpg"),
                    # f'{frame_num}.jpg'),
                    frame,
                )

    cap.release()
    cv2.destroyAllWindows()

if count_only:
    print(f"{nf} videos will generate {total_pic} pics({interval}s/pic)")
