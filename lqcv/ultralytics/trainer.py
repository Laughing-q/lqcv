from ultralytics.models.yolo import detect
from ultralytics.utils import colorstr
from .dataset import NBYOLODataset


class NBTrainer(detect.DetectionTrainer):
    def build_dataset(self, img_path, mode='train', batch=None):
        from ultralytics.utils.torch_utils import de_parallel
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        cfg = self.args
        return NBYOLODataset(
            img_path=img_path,
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment=mode == "train",  # augmentation
            hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
            rect=cfg.rect or mode == "val",  # rectangular batches
            cache=cfg.cache or None,
            single_cls=cfg.single_cls or False,
            stride=int(gs),
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            use_segments=cfg.task == "segment",
            use_keypoints=cfg.task == "pose",
            classes=cfg.classes,
            data=self.data,
            fraction=cfg.fraction if mode == "train" else 1.0,
        )
