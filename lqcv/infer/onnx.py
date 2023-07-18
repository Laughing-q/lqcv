try:
    import onnxruntime
except:
    onnxruntime = None


class ONNXModel:
    def __init__(self, model_file, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]):
        """ONNXRuntime inference.

        Args:
            model_file (str): The model file.
            providers (list[str]): The backend providers of onnxruntime.
        """
        self.model_file = model_file
        self.session = onnxruntime.InferenceSession(self.model_file, providers=providers)
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        input_name = input_cfg.name
        self.input_shape = input_shape
        outputs = self.session.get_outputs()
        output_names = []
        for out in outputs:
            output_names.append(out.name)
        self.input_name = input_name
        self.output_names = output_names

    def __call__(self, input_dict):
        """ONNXRuntime inference.

        Args:
            input_dict (dict): A dict of model inputs.

        Returns:
            output (list): A list of outputs.

        """
        assert isinstance(input_dict, dict)
        assert (
            list(input_dict.keys()) == self.input_names
        ), f"Wrong names! Expected {self.input_names} but got {list(input_dict.keys())}"
        output = self.session.run(self.output_names, input_dict)
        return output
