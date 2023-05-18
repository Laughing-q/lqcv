import shutil
import os
import tqdm


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
    less_names = [os.path.splitext(i)[0] for i in os.listdir(less_dir)]
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

if __name__ == "__main__":
    # names = find_extra_names(
    #     "/home/laughing/codes/lqcv/test/more", 
    #     "/home/laughing/codes/lqcv/test/less",
    #     reverse=False,
    # )
    # print(names)
    remove_extra_files("test/more/", "test/less/", "test/target", reverse=True)
