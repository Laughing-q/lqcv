import shutil
import os
import tqdm
import random
import glob


def find_extra_names(more_dir, less_dir, reverse=False, pure=False):
    """Assuming there are two directories have some files with the same names(common files)
    but one dir got extra files, this function is to find these extra files.

    Args:
        more_dir (str): The directory that includes more files, common files + extra files.
        less_dir (str): The directory that includes less files, common files.
        reverse (bool): This function finds these extra files by default(reverse=False),
            and it finds the common files if reverse=True.
        pure (bool): Wether to return the pure names(without suffix) or return the full names in `more_dir`,
            default: False.
    Returns:
        return_names (list): The names of these files, without suffix if pure=True.
    """
    more_names = os.listdir(more_dir)
    less_names = set([os.path.splitext(i)[0] for i in os.listdir(less_dir)])
    return_names = []
    for n in tqdm.tqdm(more_names, total=len(more_names)):
        name = os.path.splitext(n)[0]
        if reverse and name in less_names:  # if reverse then find the common files.
            return_names.append(name if pure else n)
        elif (not reverse) and (name not in less_names):  # if not reverse, then find the extra files.
            return_names.append(name if pure else n)
    return return_names


def remove_extra_files(more_dir, less_dir, target_dir=None, reverse=False):
    """Remove the extra files that in `more_dir` but not included in `less_dir`.

    Args:
        more_dir (str): The directory that includes more files, common files + extra files.
        less_dir (str): The directory that includes less files, common files.
        target_dir (str, optional): The target directory that receive extra files from `more_dir`,
            this function would directly remove extra files if `target_dir` is None(by default).
        reverse (bool): This function removes/moves these extra files by default(reverse=False),
            and it removes/moves the common files if reverse=True.
    """
    names = find_extra_names(more_dir=more_dir, less_dir=less_dir, reverse=reverse)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
    for name in names:
        file = os.path.join(more_dir, name)
        shutil.move(file, target_dir) if target_dir else os.remove(file)


def split_images(data_dir, ratio=0.8):
    """Split the data to train/val set.

    Args:
        data_dir (str): Data directory.
        ratio (float): The split ratio for train set, (1 - ratio) for val set.
    """
    files = glob.glob(os.path.join(data_dir, "*"))
    assert len(files), f"There's no files in {data_dir}"
    random.shuffle(files)
    num_train = int(len(files) * ratio)

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    for d in (train_dir, val_dir):
        os.makedirs(d)

    for i, file in tqdm.tqdm(enumerate(files), total=len(files)):
        if i < num_train:
            shutil.move(file, train_dir)
        else:
            shutil.move(file, val_dir)


def split_images_labels(data_dir, ratio=0.8):
    """Split the data to train/val set.
    Assuming the data structure is like:
        --data_dir
            --images
            --labels
    Args:
        data_dir (str): Data directory.
        ratio (float): The split ratio for train set, (1 - ratio) for val set.
    """
    image_dir = os.path.join(data_dir, "images")
    assert os.path.exists(image_dir), f"Invalid directory `{image_dir}`."
    # split images
    split_images(image_dir, ratio)

    # split labels
    label_dir = os.path.join(data_dir, "labels")
    train_dir = os.path.join(label_dir, "train")
    val_dir = os.path.join(label_dir, "val")
    remove_extra_files(more_dir=label_dir, less_dir=f"{image_dir}/train", target_dir=train_dir, reverse=True)
    remove_extra_files(more_dir=label_dir, less_dir=f"{image_dir}/val", target_dir=val_dir, reverse=True)


def get_files(root, suffix="", decs="", shuffle=False, max_num=None):
    """
    Get a list of files in the specified root directory with the given suffix.

    Args:
        root (str): The root directory to search for files.
        suffix (str): The suffix of the files to search for. Defaults to an empty string.
            Example: ".jpg".
        decs (str): The description of the tqdm iterator. Defaults to an empty string.
        max_num (int): The maximum number of files to return. Defaults to None.

    Returns:
        tqdm: A tqdm iterator over the list of files with the specified suffix.
    """
    from pathlib import Path
    from tqdm import tqdm

    files = list(Path(root).glob(f"*{suffix}"))
    if shuffle:
        random.shuffle(files)
    if max_num is not None:
        assert isinstance(max_num, int), f"Expected `max_num` should be an integer, but got {type(max_num)}."
        files = files[:max_num]
    return tqdm(files, desc=decs)


if __name__ == "__main__":
    # names = find_extra_names(
    #     "/home/laughing/codes/lqcv/test/more",
    #     "/home/laughing/codes/lqcv/test/less",
    #     reverse=False,
    # )
    # print(names)
    remove_extra_files("test/more/", "test/less/", "test/target", reverse=True)
