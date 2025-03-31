try:
    import openvino as ov
except:
    ov = None
from pathlib import Path


class OVModel:
    """A wrapper class for OpenVINO model inference.

    This class initializes and compiles an OpenVINO model for inference
    using the specified model file. It supports automatic device selection
    and latency-optimized performance.
    """

    def __init__(self, model_file) -> None:
        """Initializes the OpenVINO model.

        Args:
            model_file (str or Path): Path to the OpenVINO model file (.xml).
        """
        core = ov.Core()
        model_file = Path(model_file)
        ov_model = core.read_model(model=str(model_file), weights=model_file.with_suffix(".bin"))
        if ov_model.get_parameters()[0].get_layout().empty:
            ov_model.get_parameters()[0].set_layout(ov.Layout("NCHW"))

        self.ov_model = core.compile_model(
            ov_model,
            device_name="AUTO",  # AUTO selects best available device, do not modify
            config={"PERFORMANCE_HINT": "LATENCY"},
        )

    def __call__(self, input):
        """Performs inference on the input data.

        Args:
            input (np.ndarray): Input data for the model.

        Returns:
            list: A list of output values from the model inference.
        """
        return list(self.ov_model(input).values())
