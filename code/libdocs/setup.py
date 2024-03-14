import os
from typing import AnyStr

from setuptools import find_packages, setup


def read(fname: str) -> AnyStr:
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="libdocs",
    version="0.1",
    packages=find_packages(),
    # install_requires=['requirements.txt'],
)
