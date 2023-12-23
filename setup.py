from pathlib import Path
import setuptools
import re


def parse_requirements(file_path: Path):
    """
    Parse a requirements.txt file, ignoring lines that start with '#' and any text after '#'.

    Args:
        file_path (str | Path): Path to the requirements.txt file.

    Returns:
        (List[str]): List of parsed requirements.
    """

    requirements = []
    for line in Path(file_path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            requirements.append(line.split('#')[0].strip())  # ignore inline comments

    return requirements

with open("lqcv/__init__.py", "r") as f:
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        f.read(), re.MULTILINE
    ).group(1)


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


setuptools.setup(
    name="lqcv",
    version=version,
    author="Laughing-q",
    python_requires=">=3.6",
    long_description=long_description,
    classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent"],
    # cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    packages=setuptools.find_packages(),
    install_requires=parse_requirements(Path(__file__).resolve().parent / 'requirements.txt'),
)
