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
    from setuptools import setup, find_packages
    import runpy

    ldict = runpy.run_path(op.join(op.abspath(op.dirname(__file__)),
                                   'niworkflows', 'info.py'))

    setup(
        name=PACKAGE_NAME,
        version=ldict['__version__'],
        description=ldict['__description__'],
        long_description=ldict['__longdesc__'],
        author=ldict['__author__'],
        author_email=ldict['__email__'],
        maintainer=ldict['__maintainer__'],
        maintainer_email=ldict['__email__'],
        license=ldict['__license__'],
        url=ldict['URL'],
        download_url=ldict['DOWNLOAD_URL'],
        classifiers=ldict['CLASSIFIERS'],
        packages=find_packages(exclude=['*.tests']),
        zip_safe=False,
        # Dependencies handling
        setup_requires=ldict['SETUP_REQUIRES'],
        install_requires=list(set(ldict['REQUIRES'])),
        dependency_links=ldict['LINKS_REQUIRES'],
        tests_require=ldict['TESTS_REQUIRES'],
        extras_require=ldict['EXTRA_REQUIRES'],
        # Data
        package_data={'niworkflows': ['data/t1-mni_registration*.json', 'viz/*.tpl',
                                      'nipype/pipeline/engine/report_template.html',
                                      'nipype/external/d3.js']},
        include_package_data=True,
    )

if __name__ == '__main__':
    main()
