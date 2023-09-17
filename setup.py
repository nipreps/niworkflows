#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""niworkflows setup script."""
import sys
from setuptools import setup
import versioneer

# Use setup_requires to let setuptools complain if it's too old for a feature we need
# 40.8.0 includes license_file, reducing MANIFEST.in requirements
#
# Keep the installation version synchronized with pyproject.toml
SETUP_REQUIRES = ["setuptools >= 40.8.0"]

# This enables setuptools to install wheel on-the-fly
if "bdist_wheel" in sys.argv:
    SETUP_REQUIRES += ["wheel"]

if __name__ == "__main__":
    # Note that "name" is used by GitHub to determine what repository provides a package
    # in building its dependency graph.
    setup(
        name="niworkflows",
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        setup_requires=SETUP_REQUIRES,
    )
