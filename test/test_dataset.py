from lqcv.ultralytics.dataset import LQDataset
from lqcv.utils.plot import cv2_imshow, plot_images
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import DEFAULT_CFG as cfg
import cv2


def test_lqdataset(img_path, data, mode="train"):
    dataset = LQDataset(
        img_path=img_path,
        imgsz=640,
        batch_size=2,
        augment=mode == "train",
        hyp=cfg,
        rect=cfg.rect or mode == "val",
        stride=32,
        pad=0.0 if mode == "train" else 0.5,
        task='detect',
        data=data,
    )
    for label in dataset:
        plot_im = plot_images(
            images=label["img"][None],
            batch_idx=label["batch_idx"],
            cls=label["cls"].squeeze(-1),
            bboxes=label["bboxes"],
            paths=label["im_file"],
            fname="sample.jpg",
        )
        cv2_imshow(cv2.cvtColor(plot_im, cv2.COLOR_BGR2RGB))



if __name__ == "__main__":
    extra_args = dict(
        neg_dir="/d/dataset/audio/HDTF_DATA/RD25_images_lq/Radio1_0/",
        # bg_dir="/d/dataset/长沙/car/final/20230206/images/train",
        area_thr=0.3,
        fg_dir="/d/dataset/smoke_fire/smoke_fire/total/masks",
    )
    for k, v in extra_args.items():
        cfg.__setattr__(k, v)
    # data = check_det_dataset("/d/dataset/smoke_fire/smoke_fire/total/smoke_fire.yaml")
    data = check_det_dataset("/d/dataset/smoke_fire/smoke_fire/total/smoke_fire_bg.yaml")
    test_lqdataset(data["train"], data)
