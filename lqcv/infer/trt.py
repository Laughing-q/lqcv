import torch
import numpy as np
from collections import OrderedDict, namedtuple
try:
    import tensorrt as trt
except:
    trt = None


class TRTModel:
    def __init__(self, model_file) -> None:
        """Tensorrt engine inference.
        Input ndoe:`images`.
        Output ndoe:`output`.
        """
        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        logger = trt.Logger(trt.Logger.INFO)
        with open(model_file, "rb") as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        self.context = model.create_execution_context()
        self.bindings = OrderedDict()
        self.half = False
        self.input_names = []
        self.output_names = []
        is_trt10 = not hasattr(model, "num_bindings")
        num = range(model.num_io_tensors) if is_trt10 else range(model.num_bindings)
        for i in num:
            if is_trt10:
                name = model.get_tensor_name(i)
                dtype = trt.nptype(model.get_tensor_dtype(name))
                is_input = model.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                if is_input:
                    self.input_names.append(name)
                    if -1 in tuple(model.get_tensor_shape(name)):
                        self.context.set_input_shape(name, tuple(model.get_tensor_profile_shape(name, 0)[1]))
                    if dtype == np.float16:
                        self.half = True
                else:
                    self.output_names.append(name)
                shape = tuple(self.context.get_tensor_shape(name))
            else:  # TensorRT < 10.0
                name = model.get_binding_name(i)
                dtype = trt.nptype(model.get_binding_dtype(i))
                is_input = model.binding_is_input(i)
                if model.binding_is_input(i):
                    self.input_names.append(name)
                    if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                        self.context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[1]))
                    if dtype == np.float16:
                        self.half = True
                else:
                    self.output_names.append(name)
                shape = tuple(self.context.get_binding_shape(i))
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).cuda()
            self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())

    def __call__(self, input_dict):
        """Tensorrt inference.

        Args:
            input_dict (dict): A dict of model inputs.

        Returns:
            output (list): A list of outputs.
            
        """
        assert isinstance(input_dict, dict)
        assert list(input_dict.keys()) == self.input_names,\
                f"Wrong names! Expected {self.input_names} but got {list(input_dict.keys())}"

        for k, v in input_dict.items():
            self.binding_addrs[k] = int(v.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        y = [self.bindings[x].data.clone() for x in sorted(self.output_names)]
        return y


if __name__ == "__main__":
    model = TRTModel("weights/latest_net_G.engine")
    print(model.half, model.input_names, model.output_names)
