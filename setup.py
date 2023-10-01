import setuptools
from setuptools import setup
from os import path

ext_modules = []

here = path.abspath(path.dirname(__file__))
requires_list = []
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        requires_list.append(str(line))

setup(
    name="ss",
    version="0.1",
    description="Implementation of Sinkhorn Step in JAX",
    author="An T. Le",
    author_email="an@robot-learning.de",
    packages=setuptools.find_packages(),
    install_requires=requires_list,
)
