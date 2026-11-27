"""Minimal packaging scaffold for diffusionPack.

The first DDPM components are preliminary and under review.
"""

from setuptools import find_packages, setup

setup(
    name="diffusionpack",
    version="0.0.1",
    description="An equation-first research package for diffusion and generative models",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["torch>=2.0", "numpy>=1.22"],
    include_package_data=True,
)
