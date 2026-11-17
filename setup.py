"""Minimal packaging scaffold for diffusionPack.

The research algorithms are intentionally not implemented at this stage.
"""

from setuptools import find_packages, setup

setup(
    name="diffusionpack",
    version="0.0.1",
    description="An equation-first research package for diffusion and generative models",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[],
    include_package_data=True,
)
