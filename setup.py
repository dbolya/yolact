import os
from setuptools import setup, find_packages

from yolact import __version__


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname), "rb") as fid:
        return fid.read().decode("utf-8")


req = read("requirements.txt").splitlines()
dev_req = read("requirements-dev.txt").splitlines()[2:]
requirements = req + ["setuptools"]


setup(
    name='yolact',
    version=__version__,
    author="Psycle Research",
    description="Fork of yolact",
    url="https://github.com/PsycleResearch/yolact",
    packages=find_packages(exclude=["tests"]),
    install_requires=requirements,
    extras_require={
        'dev': dev_req
    },
    python_requires=">=3.6"
)
