#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2016-07-21 13:49:24
""" fmriprep setup script """
import os
import sys

from niworkflows import (__version__, __email__, __url__, __packagename__, __license__,
                         __description__, __longdesc__, __maintainer__, __author__)


REQ_LINKS = []
with open('requirements.txt', 'r') as rfile:
    REQUIREMENTS = [line.strip() for line in rfile.readlines()]

for i, req in enumerate(REQUIREMENTS):
    if req.startswith('-e'):
        REQUIREMENTS[i] = req.split('=')[1]
        REQ_LINKS.append(req.split()[1])

if REQUIREMENTS is None:
    REQUIREMENTS = []

def main():
    """ Install entry-point """
    from glob import glob
    from setuptools import setup

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
        packages=['niworkflows', 'niworkflows.anat', 'niworkflows.func', 'niworkflows.dwi', 'niworkflows.common'],
        # entry_points={'console_scripts': ['niworkflows=niworkflows.run_workflow:main',]},
        # package_data={'fmriprep': ['data/*']},
        install_requires=REQUIREMENTS,
        dependency_links=REQ_LINKS,
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
