#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
These pipelines are developed by the Poldrack lab at Stanford University
(https://poldracklab.stanford.edu/) for use at
the Center for Reproducible Neuroscience (http://reproducibility.stanford.edu/),
as well as for open-source software distribution.
"""
from __future__ import absolute_import, division, print_function
import datetime


__version__ = '0.0.3a4'

__packagename__ = 'niworkflows'
__author__ = 'The CRN developers'
__copyright__ = 'Copyright {}, Center for Reproducible Neuroscience, Stanford University'.format(
    datetime.datetime.now().year)
__credits__ = ['Oscar Esteban', 'Ross Blair', 'Shoshana L. Berleant', 'Chris F. Gorgolewski',
               'Russell A. Poldrack']
__license__ = '3-clause BSD'
__maintainer__ = 'Oscar Esteban'
__email__ = 'crn.poldracklab@gmail.com'
__status__ = 'Prototype'

__description__ = """NIworkflows provides processing workflows for magnetic resonance images
of the brain."""
__longdesc__ = """
NIworkflows is a selection of image processing workflows for magnetic resonance images
of the brain. It is designed to provide an easily accessible, state-of-the-art interface that is robust
to differences in scan acquisition protocols and that requires minimal user input.
This open-source neuroimaging data processing tool is being developed as a part of the
MRI image analysis and reproducibility platform offered by the
CRN.
"""

URL = 'https://github.com/poldracklab/{}'.format(__packagename__)
DOWNLOAD_URL = ('https://pypi.python.org/packages/source/n/niworkflows/'
                'niworkflows-{}.tar.gz'.format(__version__))
CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Image Recognition',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3.5',
]


REQUIRES = [
    'future',
]

LINKS_REQUIRES = [
    'git+https://github.com/oesteban/nipype.git@master#egg=nipype'
]

TESTS_REQUIRES = [
    "mock",
    "codecov"
]

EXTRA_REQUIRES = {
    'doc': ['sphinx'],
    'tests': TESTS_REQUIRES,
    'duecredit': ['duecredit']
}

# Enable a handle to install all extra dependencies at once
EXTRA_REQUIRES['all'] = [val for _, val in list(EXTRA_REQUIRES.items())]
