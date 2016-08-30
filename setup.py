#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2016-08-29 18:29:24
""" fmriprep setup script """
from __future__ import absolute_import
import os
import sys

def main():
    """ Install entry-point """
    from glob import glob
    import inspect
    from setuptools import setup, find_packages

    this_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    # Read vars from info file
    module_file =os.path.join(this_path, 'niworkflows', 'info.py')
    with open(module_file) as fvars:
        exec(compile(fvars.read(), module_file, 'exec'))

    setup(
        name=__packagename__,
        version=__version__,
        description=__description__,
        long_description=__longdesc__,
        author=__author__,
        author_email=__email__,
        email=__email__,
        maintainer=__maintainer__,
        maintainer_email=__email__,
        url=__url__,
        download_url='https://pypi.python.org/packages/source/n/niworkflows/'
                     'niworkflows-{}.tar.gz'.format(__version__),
        license=__license__,
        packages=find_packages(),
        install_requires=['nipype'],
        zip_safe=False,
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering',
            'License :: OSI Approved :: BSD License',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.5'
        ],
    )

if __name__ == '__main__':
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(local_path)
    sys.path.insert(0, local_path)

    main()
