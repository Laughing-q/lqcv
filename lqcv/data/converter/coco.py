from .base import BaseConverter
from pathlib import Path
import json


class COCOConverrter(BaseConverter):
    def __init__(self, json_file, img_dir=None) -> None:
        """COCOConverter.

        Args:
            json_file (str): The json file of coco.
            img_dir (str | optional): Image directory,
                if it's None then assume the structure is like the following example:
                    root/
                    ├── images
                    ├── json_file
        """
        if img_dir is None:
            img_dir = Path(json_file).parent / "images"
            assert img_dir.exists(), f"The directory '{str(img_dir)}' does not exist, please pass `img_dir` arg."
        super().__init__(json_file, img_dir=img_dir)

    def read_labels(self, json_file):
        with open(json_file, "r") as f:
            anno = json.load(f)
        print()
