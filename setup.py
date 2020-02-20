from setuptools import setup, find_packages

setup(name='imgclass',
      version="0.1.0",
      description='Simple image classification',
      author='Satchel Grant',
      author_email='grantsrb@stanford.edu',
      url='https://github.com/baccuslab/imgclass.git',
      install_requires=[i.strip() for i in open("requirements.txt").readlines()],
      long_description='''
          This package contains methods, classes, and model
          architectures used for simple image classificaion.
          ''',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: MacOS :: MacOS X :: Ubuntu',
          'Topic :: Scientific/Engineering :: Information Analysis'],
      packages=find_packages(),
)
