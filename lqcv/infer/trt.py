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
        for i in range(model.num_bindings):
            name = model.get_binding_name(i)
            dtype = trt.nptype(model.get_binding_dtype(i))
            # dtype = np.float16 if self.half else np.float32
            if model.binding_is_input(i):
                self.input_names.append(name)
                if dtype == np.float16:
                    self.half = True
            else:  # output
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
        y = [self.bindings[x].data for x in sorted(self.output_names)]
        return y


if __name__ == "__main__":
    model = TRTModel("weights/latest_net_G.engine")
    print(model.half, model.input_names, model.output_names)
