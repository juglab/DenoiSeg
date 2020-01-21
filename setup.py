from __future__ import absolute_import
from setuptools import setup, find_packages
from os import path

_dir = path.abspath(path.dirname(__file__))

with open(path.join(_dir,'noise2seg','version.py')) as f:
    exec(f.read())

with open(path.join(_dir,'README.md')) as f:
    long_description = f.read()


setup(name='noise2seg',
      version=__version__,
      description='Noise2Seg',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/juglab/Noise2Seg/',
      author='Mangal Prakash, Tim-Oliver Buchholz',
      author_email='prakash@mpi-cbg.de, tibuch@mpi-cbg.de',
      license='BSD 3-Clause License',
      packages=find_packages(),

      project_urls={
          'Repository': 'https://github.com/juglab/Noise2Seg/',
      },

      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
          'License :: OSI Approved :: BSD License',

          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
      ],

      install_requires=[
	      "n2v",
	      "numpy",
          "scipy",
          "matplotlib",
          "six",
          "keras>=2.2.4,<2.3.0",
          "tifffile",
          "tqdm",
          "pathlib2;python_version<'3'",
          "backports.tempfile;python_version<'3.4'",
          "csbdeep>=0.4.0,<0.5.0"
      ]
      )
