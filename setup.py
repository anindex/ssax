import setuptools
from setuptools import setup

setup(
    name="ss",
    version="0.1",
    description="Implementation of Sinkhorn Step in JAX",
    author="An T. Le",
    author_email="an@robot-learning.de",
    packages=setuptools.find_packages(),
    install_requires=[
        "wandb_plot @ git+https://github.com/danielpalen/wandb_plot.git",
    ],
)
