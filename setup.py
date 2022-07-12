import os
import setuptools
from setuptools import find_packages


def read_requirements():
    build_dir = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(build_dir, "yolact", "requirements.txt")) as f:
        return f.read().splitlines()


def packages():
    return find_packages(
        exclude=["*.__pycache__.*"]
    )



setuptools.setup(
    name="yolact",
    version="0.0.1",
    author="",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/dbolya/yolact",
    packages=packages(),
    python_requires=">=3.6",
    install_requires=read_requirements(),
    include_package_data=True,
    package_data={'': ['yolact/data/*']},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ]
)
