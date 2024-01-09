from pathlib import Path
import shutil


def export_yolov8(weight, onnx_dir="./", format="onnx", imgsz=[384, 640], pt_dir=None, check_dim=False):
    """Export yolov8 models to specific path and name. 

        The logic in this function is based on the output file structure of yolov8 training, 
    the best.pt and args.yaml inside the directory is needed.

    Args:
        weight (str | pathlib): The original pt file, usually it's the best.pt.
        save_dir (str | pathlib): The directory to put the output exported model.
        format (str): Export format.
        imgsz (List | tuple): Image size, (height, width).
        pt_dir (Optional | str): Whether to copy the original pt file to the pt_dir 
            with exactly the same name.
        check_dim (bool): Check the dimension of the output.

    Returns:
        save_path (str): The final path of exported model.
    """
    from ultralytics.utils import yaml_load
    from ultralytics import YOLO

    weight = Path(weight)
    assert weight.exists(), "The weight is NOT found!"
    arg_file = weight.parent.parent / "args.yaml"
    assert arg_file.exists(), "The arg file is NOT found!"
    args = yaml_load(arg_file)

    # get the model type, and model name from the name of data.yaml
    model_type = Path(args["model"]).stem[-1]
    model_name = Path(args["data"]).stem
    project_name = Path(args["name"])
    # NOTE: get the date, as the format of project_name is `model_name`+`date`.
    # Also the model_name is same as the name of data.yaml.
    date = project_name.replace(model_name, "")
    # upper the first letter
    model_name = model_name[0].upper() + model_name[1:]
    # export model
    model = YOLO(weight)
    file_path = model.export(format=format, imgsz=imgsz, opset=12)

    h, w = imgsz
    output_name = f"{model_name}_{h}x{w}{model_type}_{date}"
    output = output_name + str(Path(file_path).suffix)
    save_path = str(Path(onnx_dir) / output)
    shutil.move(file_path, save_path)
    if pt_dir is not None and Path(pt_dir).exists():
        shutil.copyfile(weight, str(Path(pt_dir) / f"{output_name}.pt"))

    if check_dim:
        import onnx
        model = onnx.load(save_path)
        current_dim = model.graph.output[0].type.tensor_type.shape.dim[1].dim_value
        expected_dim = 0
        for stride in [8, 16, 32]:
            expected_dim += (h / stride) * (w / stride)
        assert int(expected_dim) == int(current_dim), \
                f"Expected {int(expected_dim)} but got {current_dim}, you should add permute operation!"
    return output
