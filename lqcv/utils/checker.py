from pathlib import Path
from subprocess import check_output
from PIL import ImageFont

import os
import glob
import pkg_resources as pkg
import torch
import re
import urllib
import platform

from .general import colorstr, make_divisible

def try_except(func):
    # try-except function. Usage: @try_except decorator
    def handler(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(e)

    return handler

def is_writeable(dir, test=False):
    # Return True if directory has write permissions, test opening a file with write permissions if test=True
    if test:  # method 1
        file = Path(dir) / "tmp.txt"
        try:
            with open(file, "w"):  # open file with write permissions
                pass
            file.unlink()  # remove file
            return True
        except IOError:
            return False
    else:  # method 2
        return os.access(dir, os.R_OK)  # possible issues on Windows


def is_docker():
    # Is environment a Docker container?
    return Path("/workspace").exists()  # or Path('/.dockerenv').exists()

def is_pip():
    # Is file in a pip package?
    return "site-packages" in Path(__file__).resolve().parts


def is_ascii(s=""):
    # Is string composed of all ASCII (no UTF) characters? (note str().isascii() introduced in python 3.7)
    s = str(s)  # convert list, tuple, None, etc. to str
    return len(s.encode().decode("ascii", "ignore")) == len(s)


def is_chinese(s="人工智能"):
    # Is string composed of any Chinese characters?
    return re.search("[\u4e00-\u9fff]", s)


def emojis(str=""):
    # Return platform-dependent emoji-safe version of string
    return (
        str.encode().decode("ascii", "ignore")
        if platform.system() == "Windows"
        else str
    )


def file_size(path):
    # Return file/dir size (MB)
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / 1e6
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob("**/*") if f.is_file()) / 1e6
    else:
        return 0.0


def check_online():
    # Check internet connectivity
    import socket

    try:
        socket.create_connection(("1.1.1.1", 443), 5)  # check host accessibility
        return True
    except OSError:
        return False


@try_except
def check_git_status():
    # Recommend 'git pull' if code is out of date
    msg = ", for updates see https://github.com/ultralytics/yolov5"
    print(colorstr("github: "), end="")
    assert Path(".git").exists(), "skipping check (not a git repository)" + msg
    assert not is_docker(), "skipping check (Docker image)" + msg
    assert check_online(), "skipping check (offline)" + msg

    cmd = "git fetch && git config --get remote.origin.url"
    url = (
        check_output(cmd, shell=True, timeout=5).decode().strip().rstrip(".git")
    )  # git fetch
    branch = (
        check_output("git rev-parse --abbrev-ref HEAD", shell=True).decode().strip()
    )  # checked out
    n = int(
        check_output(f"git rev-list {branch}..origin/master --count", shell=True)
    )  # commits behind
    if n > 0:
        s = f"⚠️ YOLOv5 is out of date by {n} commit{'s' * (n > 1)}. Use `git pull` or `git clone {url}` to update."
    else:
        s = f"up to date with {url} ✅"
    print(emojis(s))  # emoji-safe


def check_python(minimum="3.6.2"):
    # Check current python version vs. required python version
    check_version(platform.python_version(), minimum, name="Python ")


def check_version(
    current="0.0.0",
    minimum="0.0.0",
    name="version ",
    pinned=False,
    hard=False,
    verbose=False,
):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    s = f"{name}{minimum} required by YOLOv5, but {name}{current} is currently installed"  # string
    if hard:
        assert result, s  # assert min requirements met
    if verbose and not result:
        import logging
        logging.warning(s)
    return result


@try_except
def check_requirements(
    requirements="requirements.txt", exclude=(), install=True
):
    # Check installed dependencies meet requirements (pass *.txt file or list of packages)
    prefix = colorstr("red", "bold", "requirements:")
    check_python()  # check python version
    if isinstance(requirements, (str, Path)):  # requirements.txt file
        file = Path(requirements)
        assert file.exists(), f"{prefix} {file.resolve()} not found, check failed."
        requirements = [
            f"{x.name}{x.specifier}"
            for x in pkg.parse_requirements(file.open())
            if x.name not in exclude
        ]
    else:  # list or tuple of packages
        requirements = [x for x in requirements if x not in exclude]

    n = 0  # number of packages updates
    for r in requirements:
        try:
            pkg.require(r)
        except Exception as e:  # DistributionNotFound or VersionConflict if requirements not met
            s = f"{prefix} {r} not found and is required by YOLOv5"
            if install:
                print(f"{s}, attempting auto-update...")
                try:
                    assert check_online(), f"'pip install {r}' skipped (offline)"
                    print(check_output(f"pip install '{r}'", shell=True).decode())
                    n += 1
                except Exception as e:
                    print(f"{prefix} {e}")
            else:
                print(f"{s}. Please install and rerun your command.")

    if n:  # if packages updated
        source = file.resolve() if "file" in locals() else requirements
        s = (
            f"{prefix} {n} package{'s' * (n > 1)} updated per {source}\n"
            f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
        )
        print(emojis(s))


def check_img_size(imgsz, s=32, floor=0):
    # Verify image size is a multiple of stride s in each dimension
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        print(
            f"WARNING: --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}"
        )
    return new_size


def check_suffix(file="yolov5s.pt", suffix=(".pt",), msg=""):
    # Check file(s) for acceptable suffix
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"


def check_yaml(file, suffix=(".yaml", ".yml")):
    # Search/download YAML file (if necessary) and return path, checking suffix
    return check_file(file, suffix)


def check_file(file, suffix=""):
    # Search/download file (if necessary) and return path
    check_suffix(file, suffix)  # optional
    file = str(file)  # convert to str()
    if Path(file).is_file() or file == "":  # exists
        return file
    elif file.startswith(("http:/", "https:/")):  # download
        url = str(Path(file)).replace(":/", "://")  # Pathlib turns :// -> :/
        file = Path(
            urllib.parse.unquote(file).split("?")[0]
        ).name  # '%2F' to '/', split https://url.com/file.txt?auth
        print(f"Downloading {url} to {file}...")
        torch.hub.download_url_to_file(url, file)
        assert (
            Path(file).exists() and Path(file).stat().st_size > 0
        ), f"File download failed: {url}"  # check
        return file
    else:  # search
        files = []
        for d in "data", "models", "utils":  # search directories
            files.extend(
                glob.glob(str(d / "**" / file), recursive=True)
            )  # find file
        assert len(files), f"File not found: {file}"  # assert file was found
        assert (
            len(files) == 1
        ), f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        return files[0]  # return file

def check_dict(dict1, dict2):
    """This func will check the keys of dict2 in dict1 or not."""
    unexpected = []
    keys = list(dict1.keys())
    for k, _ in dict2.items():
        if k not in keys:
            unexpected.append(k)
    return unexpected
