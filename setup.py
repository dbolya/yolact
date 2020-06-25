from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
   name='yolact',
   version='1.0.0',
   packages=['yolact'],    
   install_requires=required + ['dcnv2 @ git+https://github.com/CharlesShang/DCNv2@master#egg=dcnv2']
)
