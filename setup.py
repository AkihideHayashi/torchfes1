"""TorchFES."""
from setuptools import setup, find_packages

setup(
    name="torchfes",
    version="0.0.0",
    install_requires=["torch", "h5py"],
    packages=find_packages(),
)
