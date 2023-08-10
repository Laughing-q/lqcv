try:
    from ncnn_vulkan import ncnn
    # import ncnn
except:
    ncnn = None

from pathlib import Path
import numpy as np

class NCNNModel:
    """NCNN Model.

        Args:
            model_file (str): The file should be with ".param" suffix, 
                meanwhile there should be a ".bin" file in the same folder.
            use_gpu (bool): Wether to use gpu for inference.
    """
    def __init__(self, model_file, use_gpu=False):
        p = Path(model_file)
        assert p.suffix == ".param"
        model = ncnn.Net()
        # set vulkan before loading models
        model.opt.use_vulkan_compute = use_gpu

        # load param first before loading bin.
        model.load_param(model_file)
        bin_file = p.with_suffix(".bin")
        assert bin_file.exists(), "Can't find the .bin file."
        model.load_model(str(bin_file))
        self.input_names = model.input_names()
        self.output_names = model.output_names()
        self.ex = model.create_extractor()

    def __call__(self, input):
        """NCNN inference.
            
        Args:
            input (dict | np.ndarray): Inputs, with shape [C, H, W] instead of [N, C, H, W].
        """
        if isinstance(input, dict):  # multiple inputs
            assert (list(input.keys()) == self.input_names), \
                    f"Wrong names! Expected {self.input_names} but got {list(input.keys())}"
            for k, v in input.items():
                self.ex.input(k, ncnn.Mat(v))
        else:
            self.ex.input(self.input_names[0], ncnn.Mat(input))  # one input

        output = []
        for output_name in self.output_names:
            mat_out = ncnn.Mat()
            self.ex.extract(output_name, mat_out)
            output.append(np.array(mat_out)[None])
        return output


if __name__ == "__main__":
    a = Path("realesrgan-x4plus.bin")
    print(type(a.suffix), a.with_suffix(".bin"))
