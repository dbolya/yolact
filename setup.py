import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="yolact", # Replace with your own username
    version="1.2.0",
    author="Daniel Bolya",
    author_email="author@example.com",
    description="YOLACT a real-time instance segmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dbolya/yolact",
    packages=setuptools.find_packages(),
    py_modules=['yolact','backbone','eval'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
