from setuptools import setup, find_packages
import convstack

setup(name='convstack',
      version=convstack.__version__,
      description='Demonstration of linear convolutional stacking',
      author='Satchel Grant',
      author_email='grantsrb@stanford.edu',
      url='https://github.com/baccuslab/deep-retina.git',
      install_requires=[i.strip() for i in open("requirements.txt").readlines()],
      long_description='''
          This package contains methods and model architectures used to demonstrate
          the effectiveness of linear convolutional stacking.
          ''',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: MacOS :: MacOS X :: Ubuntu',
          'Topic :: Scientific/Engineering :: Information Analysis'],
      packages=find_packages(),
)
