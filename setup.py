from setuptools import setup

setup(name='yolact',
      version='2.0.3',
      author="Psycle Research",
      description="Fork of yolact",
      url="https://github.com/PsycleResearch/yolact",
      packages=['yolact', 'yolact.utils', 'yolact.data', 'yolact.layers', 'yolact.layers.functions', 'yolact.layers.modules'],
      python_requires='>=3.6'
      )
