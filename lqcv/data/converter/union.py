from .base import BaseConverter
from lqcv.bbox import Boxes


class UnionConverter(BaseConverter):
    """UnionConverter merges labels from multiple BaseConverter instances."""

    def __init__(self, converters):
        """
        Initialize UnionConverter with a list of converters.

        Args:
            converters (list): A list of BaseConverter instances.
        """
        super().__init__(label_dir=converters[0].label_dir, class_names=[])  # bypass the `label_dir` assert
        assert isinstance(converters, list), "converters must be a list of BaseConverter instances."
        for converter in converters:
            self.class_names += converter.class_names
        self.labels = self.merge(converters)
        self.updateCntInfo()
        self.format = "union"

    def read_labels(self, label_dir, chunk_size):
        """Bypass the abstract class check."""
        pass

    def merge(self, converters):
        """
        Merge labels from multiple converters.

        Args:
            converters (list): A list of BaseConverter instances.

        Returns:
            list: A list of merged labels.
        """
        labels = dict()
        cls_map = [
            {ni: self.class_names.index(name) for ni, name in enumerate(converter.class_names)}
            for converter in converters
        ]
        for i, converter in enumerate(converters):
            for label in converter.labels:
                img_name = label["img_name"]
                new_cls = [cls_map[i][c] for c in label["cls"]]
                existed_label = labels.get(img_name, None)
                if existed_label is None:
                    labels[img_name] = dict(img_name=img_name, shape=label["shape"], cls=new_cls, bbox=label["bbox"])
                else:
                    existed_label["cls"] += new_cls
                    boxes = Boxes.cat([existed_label["bbox"], label["bbox"]])
                    existed_label["bbox"] = boxes
        return list(labels.values())
