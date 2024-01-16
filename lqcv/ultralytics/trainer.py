from ultralytics.models.yolo import detect, pose, segment
from ultralytics.utils import RANK, yaml_save
from ultralytics.utils import colorstr
from copy import copy
from .dataset import LQDataset


class DetectionTrainer(detect.DetectionTrainer):
    def __init__(self, overrides=None, _callbacks=None, extra_args={}):
        super().__init__(overrides=overrides, _callbacks=_callbacks)
        # pass all extra_args to self.args
        self.val_args = copy(self.args)
        for k, v in extra_args.items():
            self.args.__setattr__(k, v)
        # save the args again with extra_args
        if RANK in (-1, 0):
            yaml_save(self.save_dir / 'args.yaml', vars(self.args))  # save run args

    def build_dataset(self, img_path, mode='train', batch=None):
        from ultralytics.utils.torch_utils import de_parallel
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        cfg = self.args
        return LQDataset(
            img_path=img_path,
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=cfg,
            rect=cfg.rect or mode == "val",
            cache=cfg.cache or None,
            single_cls=cfg.single_cls or False,
            stride=int(gs),
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            use_segments=cfg.task == "segment",
            use_keypoints=cfg.task == "pose",
            classes=cfg.classes,
            data=self.data,
            task=cfg.task,
            fraction=cfg.fraction if mode == "train" else 1.0,
        )

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.val_args), _callbacks=self.callbacks
        )

class PoseTrainer(DetectionTrainer, pose.PoseTrainer):
    def __init__(self, overrides=None, _callbacks=None, extra_args={}):
        if overrides is None:
            overrides = {}
        overrides['task'] = 'pose'
        DetectionTrainer.__init__(self, overrides=overrides, _callbacks=_callbacks, extra_args=extra_args)


class SegmentationTrainer(DetectionTrainer, segment.SegmentationTrainer):
    def __init__(self, overrides=None, _callbacks=None, extra_args={}):
        if overrides is None:
            overrides = {}
        overrides['task'] = 'segment'
        DetectionTrainer.__init__(self, overrides=overrides, _callbacks=_callbacks, extra_args=extra_args)
