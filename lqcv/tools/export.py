from pathlib import Path
import shutil
import time


def export_yolov8(weight, save_dir="./", format="onnx", imgsz=[384, 640], copy_pt=False):
    """Export yolov8 models to specific path and name. 

        The logic in this function is based on the output file structure of yolov8 training, 
    the best.pt and args.yaml inside the directory is needed.

    Args:
        weight (str | pathlib): The original pt file, usually it's the best.pt.
        save_dir (str | pathlib): The directory to put the output exported model.
        format (str): Export format.
        imgsz (List | tuple): Image size, (height, width).
        copy_pt (bool): Whether to copy the original pt file to the same save_dir 
            with exactly the same name.

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
    # upper the first letter
    model_name = model_name[0].upper() + model_name[1:]

    today = time.strftime("%Y%m%d", time.localtime())
    # export model
    model = YOLO(weight)
    file_path = model.export(format=format, imgsz=imgsz, opset=12)

    h, w = imgsz
    output_name = f"{model_name}_{h}x{w}{model_type}_{today}"
    output = output_name + str(Path(file_path).suffix)
    save_path = str(Path(save_dir) / output)
    shutil.move(file_path, save_path)
    if copy_pt:
        shutil.copyfile(weight, str(Path(save_dir) / f"{output_name}.pt"))
    return save_path
