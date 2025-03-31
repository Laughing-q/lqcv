import cv2
import torch
import numpy as np
import torch.nn as nn
from .trt import TRTModel
from .onnx import ONNXModel
from .ncnn import NCNNModel
from .openvino import OVModel


class BaseInference:
    def __init__(self, model_file, im_size=640, half=False, providers=["CUDAExecutionProvider"], ncnn_gpu=True) -> None:
        """A inference class for torch/onnx/engine models.

        Args:
            model_file (str): The model path, pt/onnx/engine/param file, param file is ncnn model.
            im_size (tuple | int): The input size for model, it should be (h, w) order if it's a tuple.
            half (bool): Half mode for torch or trt model.
            providers (list): Providers for onnx inference.
            ncnn_gpu (bool): Using gpu flag for ncnn.
        """
        self.half = half
        self.im_size = im_size if isinstance(im_size, tuple) else (im_size, im_size)
        self.engine = model_file.endswith(".engine")
        self.onnx = model_file.endswith(".onnx")
        self.ncnn = model_file.endswith(".param")
        self.ov = model_file.endswith(".xml")  # openvino
        self.providers = providers
        self.ncnn_gpu = ncnn_gpu

        self.model = self.load_model(model_file)
        self.torch = isinstance(self.model, nn.Module)
        if self.onnx or self.ncnn:
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
            model = ONNXModel(model_file, providers=self.providers)
        elif self.ncnn:
            model = NCNNModel(model_file, use_gpu=self.ncnn_gpu)
        elif self.ov:
            model = OVModel(model_file)
        else:
            model = self.load_torch_model(model_file)
        return model

    def load_torch_model(self, model_file):
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
        im = im.astype(np.float32)   # to float32 type
        if hasattr(self, "mean"):
            # NOTE: `self.mean` should be numpy type
            assert isinstance(self.mean, (np.ndarray, float))
            # assert self.mean > 1.0, "The images are unnormlized, hence the mean value should be larger than 1."
            im -= self.mean    # do normalization in RGB order
        if hasattr(self, "std"):
            # NOTE: `self.std` should be numpy type
            assert isinstance(self.std, (np.ndarray, float))
            # assert self.std > 1.0, "The images are unnormlized, hence the std value should be larger than 1."
            im /= self.std     # do normalization in RGB order
        if self.torch or self.engine:
            im = torch.from_numpy(im).cuda()   # to torch, to cuda
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

    def __call__(self, im, postprocess=True, *args, **kwargs):
        """Inference

        Args:
            im (np.ndarray | List[ndarray] | torch.Tensor): Input image.
            postprocess (bool): Whether to do postprocess.

        Returns:

        """
        im = self.preprocess(im)
        # NOTE: assuming engine/onnx inference only supports batch=1
        if self.engine:
            outputs = [self.model(self.get_input_dict(i[None], *args, **kwargs)) for i in im]
            outputs = [
                (torch.cat(output, dim=0) if len(output) > 1 else output[0])
                for output in zip(*outputs)
            ]
            # NOTE: engine postprocess could be different from torch model,
            # depends the way of how model exported.
            outputs = self.engine_postprocess(outputs) if postprocess else outputs
        elif self.onnx:
            outputs = [self.model(self.get_input_dict(i[None], *args, **kwargs)) for i in im]
            outputs = [
                np.concatenate(output, axis=0) if len(output) > 1 else output[0]
                for output in zip(*outputs)
            ]
            # NOTE: onnx postprocess could be different from torch model,
            # depends the way of how model exported.
            outputs = self.onnx_postprocess(outputs) if postprocess else outputs
        elif self.ncnn:
            # outputs = [self.model(self.get_input_dict(i, *args, **kwargs)) for i in im]
            # TODO: only support once node as input for now.
            outputs = [self.model(i, *args, **kwargs) for i in im]
            outputs = [
                np.concatenate(output, axis=0) if len(output) > 1 else output[0]
                for output in zip(*outputs)
            ]
            outputs = self.ncnn_postprocess(outputs) if postprocess else outputs
        else:
            with torch.no_grad():
                outputs = self.model(im, *args, **kwargs)
            outputs = self.torch_postprocess(outputs) if postprocess else outputs
        return outputs

    def get_input_dict(self, im):
        return {"images": im}

    def postprocess(self, outputs):
        """A unify postprocess."""
        return outputs

    def torch_postprocess(self, outputs):
        """Postprocess for torch model."""
        return self.postprocess(outputs)

    def engine_postprocess(self, outputs):
        """Postprocess for engine model.

        Args:
            outputs (list(torch.Tensor))
        """
        return self.postprocess(outputs)

    def onnx_postprocess(self, outputs):
        """Postprocess for onnx model.

        Args:
            outputs (list(np.ndarray))
        """
        return self.postprocess(outputs)

    def ncnn_postprocess(self, outputs):
        """Postprocess for ncnn model.

        Args:
            outputs (list(ncnn.Mat))
        """
        return self.postprocess(outputs)
