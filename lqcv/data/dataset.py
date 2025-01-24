from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import cv2


class ImagesDataset(Dataset):
    def __init__(self, im_dir, backend="cv2", transform=None):
        """Initializes the dataset.

        Args:
            im_dir (str): Directory containing images.
            backend (str, optional): Backend to use for image loading. Defaults to "cv2".
            transform (callable, optional): Optional transform to be applied on an image. Defaults to None.
        """
        super().__init__()
        assert backend in {"cv2", "PIL"}
        self.im_files = list(Path(im_dir).glob("*"))
        assert len(self.im_files) > 0, f"No images found in {im_dir}"
        self.backend = backend
        self.transform = transform

    def preprocess(self, im_file):
        """Preprocesses the image file.

        Args:
            im_file (str): Path to the image file.

        Returns:
            (dict): A dictionary containing the processed image, its shape, and the image file path.
        """
        if self.backend == "cv2":
            im = cv2.imread(str(im_file))
            shape = im.shape[:2]
        else:
            im = Image.open(str(im_file)).convert("RGB")
            shape = [im.height, im.width]
        im = self.transform(im) if self.transform else im
        return {"im": im, "shape": shape, "im_file": im_file}

    def __getitem__(self, index):
        """Gets the preprocessed image at the specified index.

        Args:
            index (int): Index of the image file.

        Returns:
            (dict): A dictionary containing the processed image, its shape, and the image file path.
        """
        return self.preprocess(self.im_files[index])

    def __len__(self):
        """Returns the length of the image file list for the dataset."""
        return len(self.im_files)

    def collate_fn(self, batch):
        """Collates a batch of data.

        Args:
            batch (list): List of dictionaries containing batch data.

        Returns:
            (dict): A dictionary with keys from the batch and values as lists of corresponding values from the batch.
        """
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        return {k: values[i] for i, k in enumerate(keys)}
