#!/usr/bin/env python
from distutils.core import setup

__version__ = '0.1'

setup(name = 'wTreepm',
      version = __version__,
      description = 'hacked light-weight version of Andrew Wetzels TreePM code',
      author='ChangHoon Hahn',
      author_email='hahn.changhoon@gmail.com',
      url='',
      platforms=['*nix'],
      license='GPL',
      requires = ['numpy', 'scipy'],
      provides = ['wtreepm'],
      packages = ['wtreepm', 'wtreepm.utility']
      )
