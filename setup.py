#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
""" niworkflows setup script """

PACKAGE_NAME = 'niworkflows'

def main():
    """ Install entry-point """
    from os import path as op
    from glob import glob
    from inspect import getfile, currentframe
    from setuptools import setup, find_packages
    from io import open  # pylint: disable=W0622

    this_path = op.dirname(op.abspath(getfile(currentframe())))

    # Python 3: use a locals dictionary
    # http://stackoverflow.com/a/1463370/6820620
    ldict = locals()

    # Get version and release info, which is all stored in niworkflows/info.py
    module_file = op.join(this_path, PACKAGE_NAME, 'info.py')
    with open(module_file) as infofile:
        pythoncode = [line for line in infofile.readlines() if not line.strip().startswith('#')]
        exec('\n'.join(pythoncode), globals(), ldict)

    setup(
        name=PACKAGE_NAME,
        version=ldict['__version__'],
        description=ldict['__description__'],
        long_description=ldict['__longdesc__'],
        author=ldict['__author__'],
        author_email=ldict['__email__'],
        email=ldict['__email__'],
        maintainer=ldict['__maintainer__'],
        maintainer_email=ldict['__email__'],
        license=ldict['__license__'],
        url=ldict['URL'],
        download_url=ldict['DOWNLOAD_URL'],
        classifiers=ldict['CLASSIFIERS'],
        packages=find_packages(),
        zip_safe=False,
        package_data={'niworkflows': ['data/t1-mni_registration*.json']},
        # Dependencies handling
        setup_requires=[],
        install_requires=ldict['REQUIRES'],
        dependency_links=ldict['LINKS_REQUIRES'],
        tests_require=ldict['TESTS_REQUIRES'],
        extras_require=ldict['EXTRA_REQUIRES'],
    )

if __name__ == '__main__':
    main()
