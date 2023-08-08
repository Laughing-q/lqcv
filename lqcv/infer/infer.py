import cv2
import torch
import numpy as np
import torch.nn as nn
from .trt import TRTModel
from .onnx import ONNXModel


class BaseModel:
    def __init__(self, model_file, im_size, half=False) -> None:
        """A inference class for torch/onnx/engine models.

        Args:
            model_file (str): The model path, pt/onnx/engine file.
            im_size (tuple | int): The input size for model, it should be (h, w) order if it's a tuple.
            half (bool): Half mode for torch model.
        """
        self.half = half
        self.im_size = im_size if isinstance(im_size, tuple) else (im_size, im_size)
        self.engine = model_file.endswith(".engine")
        self.onnx = model_file.endswith(".onnx")
        self.model = self.load_model(model_file)
        self.torch = isinstance(self.model, nn.Module)
        if self.onnx:
            assert self.half == False, "ONNX model is not compatible with half mode!"
        # NOTE: for torch model
        if self.torch:
            self.model.cuda()
            self.model.half() if half else self.model.float()
            self.model.eval()

    def load_model(self, model_file):
        # create model and load weights
        if self.engine:
            model = TRTModel(model_file)
            self.half = model.half
        elif self.onnx:
            model = ONNXModel(model_file)
        else:
            model = self.load_torch_model(model_file)
        return model

    def load_torch_model(self):
        raise NotImplementedError

    def preprocess(self, im):
        """Preprocess

        Args:
            im (np.ndarray | List[ndarray] | torch.Tensor): Input image.

        Returns:

        """
        # NOTE: assuming img is in CUDA with (b, 3, h, w) and divided by 255 in advance if it's a tensor
        if isinstance(im, np.ndarray):
            # numpy with (h, w, 3)
            im = self._single_preprocess(im)
        elif isinstance(im, list):
            im = np.concatenate([self._single_preprocess(i) for i in im], axis=0)
        if hasattr(self, "mean") and hasattr(self, "std"):
            # NOTE: `self.mean` and `self.std` should be numpy type
            assert isinstance(self.mean, np.ndarray) and isinstance(self.std, np.ndarray)
            im = (im - self.mean) / self.std  # do normalization in RGB order
        if self.onnx:
            im = im.astype(np.float32) / 255.0
        else:
            # NOTE: transferring to cuda first with uint8 type for faster speed.
            im = torch.from_numpy(im).cuda().float()
            im /= 255.0
        im = im.half() if self.half else im
        return im

    def _single_preprocess(self, im):
        """Process image from HWC to 1CHW, BGR to RGB.

        Args:
            im (np.ndarray): Input image.

        Returns:

        """
        assert isinstance(im, np.ndarray)
        im = self.pre_transform(im)
        im = im.transpose((2, 0, 1))[::-1][None]  # HWC to 1CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        return im

    def pre_transform(self, im):
        """Pre-transform input image before inference.

        Args:
            im (np.ndarray): Input image.

        Return: transformed imgs.
        """
        return cv2.resize(im, self.im_size[::-1])

    def __call__(self, im):
        """Inference

        Args:
            im (np.ndarray | List[ndarray] | torch.Tensor): Input image.

        Returns:

        """
        im = self.preprocess(im)
        # NOTE: assuming engine/onnx inference only supports batch=1
        if self.engine:
            outputs = [self.model(self.get_input_dict(i[None])) for i in im]
            outputs = [
                (torch.cat(output, dim=0) if len(output) > 1 else output[0]).clone()
                for output in zip(*outputs)
            ]
            # NOTE: engine postprocess could be different from torch model,
            # depends the way of how model exported.
            outputs = self.engine_postprocess(outputs)
        elif self.onnx:
            outputs = [self.model(self.get_input_dict(i[None])) for i in im]
            outputs = [
                np.concatenate(output, axis=0) if len(output) > 1 else output[0]
                for output in zip(*outputs)
            ]
            # NOTE: onnx postprocess could be different from torch model,
            # depends the way of how model exported.
            outputs = self.onnx_postprocess(outputs)
        else:
            outputs = self.torch_postprocess(self.model(im))
        return outputs

    def get_input_dict(self, im):
        return {"images": im}

    def torch_postprocess(self, outputs):
        """Postprocess for torch model."""
        return outputs

    def engine_postprocess(self, outputs):
        """Postprocess for engine model."""
        return self.postprocess(outputs)

    def onnx_postprocess(self, outputs):
        """Postprocess for onnx model."""
        return self.postprocess(outputs)
