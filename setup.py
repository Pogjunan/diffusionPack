from setuptools import find_packages, setup

setup(
    name="diffusionpack",
    version="0.0.1",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["torch>=2.3", "torchvision>=0.18", "numpy>=1.22", "pandas>=2.0", "pillow>=9", "tqdm>=4.66"],
)
